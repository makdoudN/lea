import os
import gym
import tqdm
import copy
import hydra
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from rltk.common.logger import EpochLogger
from rltk.common.utils import import_class_from_string

from envs import make_env
from runner import enjoy, Runner
from utils import ReplayBuffer, load_library

from box import Box
from pathlib import Path
from tabulate import tabulate
from omegaconf import OmegaConf
from typing import Optional
from collections import Iterable, deque
from torch.utils.tensorboard import SummaryWriter


@hydra.main(config_path="./conf/train_baseline.yaml", strict=False)
def train(conf):
    torch.set_num_threads(1)
    writer = SummaryWriter()
    L = EpochLogger(os.getcwd(), sep="\t")
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    env = make_env(conf.env.name, conf.env.loader, **conf.env.kwargs)
    env_test = make_env(conf.env.name, conf.env.loader, **conf.env.kwargs)
    conf.env.observation_nd = env.observation_space.shape[0]
    conf.env.action_nd = env.action_space.shape[0]
    if "actor_update_freq" in conf.rl.kwargs:
        conf.num_batches_old = conf.num_batches
        conf.num_batches = conf.num_batches * conf.rl.kwargs.actor_update_freq
    print(conf.pretty(resolve=True))

    models, optims = Box(), Box()
    models.actor = import_class_from_string(conf.actor.cls)(**conf.actor.kwargs)
    models.critic = import_class_from_string(conf.critic.cls)(**conf.critic.kwargs)
    targets = Box({k: copy.deepcopy(v) for k, v in models.items()})
    optims = Box(
        {
            k: import_class_from_string(conf[f"{k}_optim"].cls)(
                v.parameters(), **conf[f"{k}_optim"].kwargs
            )
            for k, v in models.items()
        }
    )
    for k, v in models.items():
        v.to(conf.device)
    for k, v in targets.items():
        v.to(conf.device)

    memory = ReplayBuffer(**conf.memory.kwargs)
    model = import_class_from_string(conf.rl.cls)(
        models=models,
        targets=targets,
        optims=optims,
        logger=L,
        **conf.rl.kwargs,
        **conf.common.rl,
    )

    runner = Runner(
        env=env,
        model=model,
        memory=memory,
        device=conf.device,
        **conf.runner.kwargs,
    )
    sample = iter(runner)

    # Learning Loop.
    for e in range(conf.num_epochs):
        info = next(sample)
        for j in range(conf.num_batches):
            batch = runner.sample(conf.batch_size)
            model.update(*batch)
        eval_score = runner.enjoy(model, env_test, n=10)
        epoch_info = L.log(
            {
                **info,
                "score": eval_score,
                "epoch": e,
                "timesteps": e * conf.runner.kwargs.horizon,
            }
        )
        print(tabulate(list(epoch_info.items()), tablefmt="fancy_grid"))
        for k, v in epoch_info.items():
            writer.add_scalar(k, v, e)

        # Checkpoints (if needed.)
        if conf.log_save_interval and e % conf.log_save_interval == 0:
            torch.save(
                model.state_dict(), os.path.join(checkpoint_dir, f"models-{e}.p")
        )

    torch.save(model.state_dict(), os.path.join(L.output_dir, "models.p"))
    OmegaConf.save(conf, os.path.join(os.getcwd(), "conf.yaml"), resolve=True)


if __name__ == "__main__":
    train()
