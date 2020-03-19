import torch
import collections
import numpy as np
import torch.nn as nn

from box import Box
from typing import Optional

from rltk.algorithms.base import BaseModel
from rltk.common.maths import polyak_averaging


class DDPG(BaseModel):
    def __init__(
        self,
        models,
        targets,
        optims: Optional[torch.optim.Optimizer],
        device: str = "cpu",
        gamma: float = 0.99,
        polyak: float = 0.005,
        batch_size: int = 128,
        action_epsilon_nz: float = 0.0,
        action_epsilon_rnd: float = 0.1,
        action_penalty_lam: float = 0.0,
        max_grad_norm: float = -1,
        logger: Optional = None,
    ):

        self.models = models
        self.optims = optims
        self.targets = targets
        self.gamma = gamma
        self.polyak = polyak
        self.device = device
        self.batch_size = batch_size
        self.action_epsilon_nz = action_epsilon_nz
        self.action_epsilon_rnd = action_epsilon_rnd
        self.action_penalty_lam = action_penalty_lam
        self.max_grad_norm = max_grad_norm
        self.logger = logger

    @torch.no_grad()
    def act(self, obses: torch.Tensor, mode: str = "train"):
        action = self.models.actor(obses).cpu().detach().numpy()
        if not mode == "train":
            return action
        action += self.action_epsilon_nz * np.random.randn(*action.shape)
        action = np.clip(action, -1, 1)
        action += np.random.binomial(1, self.action_epsilon_rnd, 1)[0] * (
            np.random.uniform(low=-1, high=1, size=action.shape) - action
        )
        return action, {}

    def update(self, obses, action, reward, mask, next_obses):
        loss_actor = self.learn_actor(obses)
        loss_critic = self.learn_critic(obses, reward, action, mask, next_obses)

        self.optims.critic.zero_grad()
        loss_critic.backward()
        if self.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.models.critic.parameters(), self.max_grad_norm, norm_type=2
            )
        self.optims.critic.step()

        self.optims.actor.zero_grad()
        loss_actor.backward()
        if self.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.models.actor.parameters(), self.max_grad_norm, norm_type=2
            )
        self.optims.actor.step()

        for k, v in self.targets.items():
            polyak_averaging(self.models[k], v, tau=self.polyak)

    def learn_critic(self, obses, reward, action, mask, next_obses):
        with torch.no_grad():
            next_action = self.targets.actor(next_obses)
            next_qfunc = self.targets.critic(next_obses, next_action)
            target = reward + self.gamma * mask * next_qfunc.squeeze(1)
        qfunc = self.models.critic(obses, action).squeeze(1)
        assert target.shape == qfunc.shape
        loss = (qfunc - target).pow(2).mean().mul(0.5)
        if self.logger is not None:
            self.logger.store(loss_critic=np.round(loss.item(), 3))
        return loss

    def learn_actor(self, obses):
        action = self.models.actor(obses)
        action_penalty = self.action_penalty_lam * action.pow(2).mean()
        loss = self.models.critic(obses, action).mean().neg()
        loss += action_penalty
        if self.logger is not None:
            self.logger.store(loss_actor=np.round(loss.item(), 3))
        return loss

    def state_dict(self):
        models = {k: v.state_dict() for k, v in self.models.items()}
        optims = {f"optims-{k}": v.state_dict() for k, v in self.optims.items()}
        targets = {f"targets-{k}": v.state_dict() for k, v in self.targets.items()}
        return {
            **models,
            **targets,
            **optims,
        }


class TD3(DDPG):
    def __init__(
        self,
        action_noise_clip: float = 0.5,
        action_noise: float = 0.2,
        actor_update_freq: int = 2,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.action_noise = action_noise
        self.action_noise_clip = action_noise_clip
        self.actor_update_freq = actor_update_freq
        self.num_updates = 0

    def update(self, obses, action, reward, mask, next_obses):
        if self.num_updates % self.actor_update_freq == 0:
            loss_actor = self.learn_actor(obses)
        loss_critic = self.learn_critic(obses, reward, action, mask, next_obses)
        if self.logger is not None:
            self.logger.store(loss_critic=np.round(loss_critic.item(), 3))

        self.optims.critic.zero_grad()
        loss_critic.backward()
        if self.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.models.critic.parameters(), self.max_grad_norm, norm_type=2
            )
        self.optims.critic.step()

        if self.num_updates % self.actor_update_freq == 0:
            self.optims.actor.zero_grad()
            loss_actor.backward()
            if self.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.models.actor.parameters(), self.max_grad_norm, norm_type=2
                )
            self.optims.actor.step()
            for k, v in self.targets.items():
                polyak_averaging(self.models[k], v, tau=self.polyak)
                if self.logger is not None:
                    self.logger.store(loss_actor=np.round(loss_actor.item(), 3))

    def learn_critic(self, obses, reward, action, mask, next_obses):
        with torch.no_grad():
            next_action = self.targets.actor(next_obses)
            #  We slighly perturbe `action_next` to enhance the robustness and decrease potential
            #  value overestimation as specify in TD3 paper.
            action_noise = next_action.data.clone().normal_(0, self.action_noise)
            action_noise = action_noise.clamp(
                -self.action_noise_clip, self.action_noise_clip
            )
            #  We assume action_space is normalized to one.
            next_action = torch.clamp(next_action + action_noise, -1, 1)
            next_qfunc = torch.min(*self.targets.critic(next_obses, next_action))
            target = reward + self.gamma * mask * next_qfunc.squeeze(1)
        qfunc_1, qfunc_2 = self.models.critic(obses, action)
        loss = (qfunc_1.squeeze(1) - target).pow(2).mean().mul(0.5) + (
            qfunc_2.squeeze(1) - target
        ).pow(2).mean().mul(0.5)
        if self.logger is not None:
            self.logger.store(loss_critic=np.round(loss.item(), 3))
        return loss

    def learn_actor(self, obses):
        action = self.models.actor(obses)
        action_penalty = self.action_penalty_lam * action.pow(2).mean()
        loss = self.models.critic(obses, action)[0].mean().neg()
        loss += action_penalty
        return loss
