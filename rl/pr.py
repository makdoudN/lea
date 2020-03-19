import random
import torch
import collections
import numpy as np
import torch.nn as nn
from rltk.common.maths import polyak_averaging

from box import Box
from typing import Optional
from .ddpg import DDPG, TD3


class DDPG_PiReuse(DDPG):
    def __init__(
            self,
            library,
            pi_reuse,
            reuse: bool = True,
            num_commit: int = 1,
            lam_pi_guidance: float = 0.001,
            use_pi_guidance: bool = False,
            use_q_guidance: bool = False,
            *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.library = library
        self.library.append(self.targets.actor)
        self.reuse = reuse
        self.pi_reuse = pi_reuse
        self.num_commit = num_commit
        self.index_commit = None
        # When the agent reuse advice it should commit to the advisor
        # during a number 'num_commit' times.
        self.in_reuse_mode = False
        self.last_n_commint = self.num_commit
        self.count_advice = np.zeros(len(self.library))

    def act(self, obses, mode: str = "train"):
        if mode == "train" and self.in_reuse_mode and self.reuse:
            with torch.no_grad():
                self.last_n_commint -= 1
                action = self.library[self.index_commit](obses).cpu().numpy()
                action = np.clip(action, -1, 1)
                # Increment the count of played advice
                self.count_advice[self.index_commit] += 1
                if self.last_n_commint == 0:
                    self.index_commit = None
                    self.in_reuse_mode = False
                    self.last_n_commint = self.num_commit
            return action, {}
        if (not self.in_reuse_mode and random.random() >= self.pi_reuse) or (
            not mode == "train" or not self.reuse
        ):
            if mode == "train":
                # Student policy is played.
                self.count_advice[-1] += 1
            return super().act(obses, mode=mode)

        if not self.reuse:
            raise ValueError()

        advice = [advisor(obses) for advisor in self.library]
        action, index = self.filter(obses, advice)
        self.in_reuse_mode = True
        self.index_commit = index
        action = np.clip(action, -1, 1)
        self.count_advice[self.index_commit] += 1
        return action, {}

    @torch.no_grad()
    def filter(self, x, advice):
        # NOTE: MERGE
        if len(x.shape) == 1:
            index = torch.argmax(
                torch.cat([self.targets.critic(x, a) for a in advice])
            ).item()
            return advice[index].detach().cpu().numpy(), index
        q = torch.stack([self.targets.critic(x, a) for a in advice])
        index = torch.argmax(q, dim=0)
        advice = torch.stack(advice)
        best_advice = advice[index.squeeze(-1), range(advice.shape[1])]
        return best_advice, index


class TD3_PiReuse(DDPG_PiReuse):
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

    @torch.no_grad()
    def filter(self, x, advice):
        if len(x.shape) == 1:
            index = torch.argmax(
                torch.cat([
                    torch.min(torch.stack(self.targets.critic(x, a)), dim=0)[0] for a in advice])
            ).item()
            return advice[index].detach().cpu().numpy(), index
        q = torch.stack([
            torch.stack(self.targets.critic(x, a)).min(0)[0] for a in advice])
        index = torch.argmax(q, dim=0)
        advice = torch.stack(advice)
        best_advice = advice[index.squeeze(-1), range(advice.shape[1])]
        return best_advice, index

    def learn_actor(self, obses):
        action = self.models.actor(obses)
        action_penalty = self.action_penalty_lam * action.pow(2).mean()
        q_sa = self.models.critic(obses, action)[0]
        loss = q_sa.mean().neg()
        loss += action_penalty
        if self.logger is not None:
            self.logger.store(loss_actor=np.round(loss.item(), 3))
        return loss

    def learn_critic(self, obses, reward, action, mask, next_obses):
        with torch.no_grad():
            next_action = self.targets.actor(next_obses)
            action_noise = next_action.data.clone().normal_(0, self.action_noise)
            action_noise = action_noise.clamp(
                -self.action_noise_clip, self.action_noise_clip
            )
            #  We assume action_space is normalized to one.
            next_action = torch.clamp(next_action + self.action_noise, -1, 1)
            next_qfunc = torch.min(*self.targets.critic(next_obses, next_action))
            target = reward + self.gamma * mask * next_qfunc.squeeze(1)
        qfunc_1, qfunc_2 = self.models.critic(obses, action)
        loss = (qfunc_1.squeeze(1) - target).pow(2).mean().mul(0.5) + (
            qfunc_2.squeeze(1) - target
        ).pow(2).mean().mul(0.5)
        if self.logger is not None:
            self.logger.store(loss_critic=np.round(loss.item(), 3))
        return loss

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


