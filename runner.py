import gym
import torch
import numpy as np

from box import Box
from typing import Optional, Callable
from collections import defaultdict, Iterable, deque

from rltk.algorithms.base import BaseRunner


class Runner(BaseRunner):
    def __init__(self, memory, random_init: int = 10000, ignore_timeout_done: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = 0
        self.memory = memory
        self.random_init = random_init
        self.ignore_timeout_done = ignore_timeout_done

    def __iter__(self) -> dict:
        env = self.env
        obses = env.reset()
        model = self.model
        score = 0
        episode_steps = 0
        memory = self.memory
        epinfo = deque(maxlen=10)
        while True:
            self.steps += 1
            episode_steps += 1
            if model is not None and len(memory) > self.random_init:
                action, info_action = model.act(
                    obses=self.process_observation(obses), mode="train"
                )
            else:
                action, info_action = env.action_space.sample(), {}
            new_obses, reward, is_done, env_info = env.step(action)
            if self.ignore_timeout_done and (episode_steps == self.env.spec.max_episode_steps):
                is_done = False
            memory.add(obses, action, reward, 1 - int(is_done), new_obses)
            obses = new_obses.copy()
            score += reward
            if is_done or episode_steps == self.env.spec.max_episode_steps:
                epinfo.append(score)
                episode_steps = 0
                score = 0
                obses = env.reset()
            if self.steps % self.horizon == 0 and len(self.memory) > self.random_init:
                yield {"training_score": np.mean(epinfo), "timesteps": self.steps}

    def sample(self, bz: int):
        (state, action, reward, done, next_state), *_ = self.memory.sample(bz)
        state = self.process_observation(state)
        next_state = self.process_observation(next_state)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.float32, device=self.device)
        return state, action, reward, done, next_state


def enjoy(
            policy,
            env: gym.Env,
            n: int = 5,
            device: str = 'cpu',
            render: bool = False,
            max_episode_steps: Optional[int] = None,
            record: bool = False,
            record_save_dir: str = "./recorded_rollout",
            record_kwargs: dict = {},
        ):
    if record:
        height = record_kwargs.get("height", 256)
        width = record_kwargs.get("width", 256)
        camera_id = record_kwargs.get("camera_id", 0)
        fps = record_kwargs.get("fps", 30)
    try:
        max_episode_steps = (
            max_episode_steps
            if max_episode_steps is not None
            else env.spec.max_episode_steps
        )
    except:
        max_episode_steps = int(9e9)
    all_score = []
    for j in range(n):
        if record:
            frames = []
        score = 0
        ob = env.reset()
        for _ in range(max_episode_steps):
            ob = torch.tensor(ob, dtype=torch.float, device=device)
            action = policy(ob)
            ob, rew, done, info = env.step(action)
            if render:
                env.render()
            if record:
                frames.append(
                    env.render(
                        mode="rgb_array",
                        height=height,
                        width=width,
                        camera_id=camera_id,
                    )
                )
            score += rew
            if done:
                all_score.append(score)
                break
        if record:
            path = Path(record_save_dir)
            path.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(os.path.join(str(path), str(j)), frames, fps=fps)
    return np.mean(all_score)



