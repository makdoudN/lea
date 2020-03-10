import random
import torch
import collections
import numpy as np
import torch.nn as nn

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


class TD3_PiReuse(TD3):
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
        raise
        # NOTE: MERGE
        if len(x.shape) == 1:
            index = torch.argmax(
                torch.cat([torch.min(self.targets.critic(x, a)) for a in advice])
            ).item()
            return advice[index].detach().cpu().numpy(), index
        q = torch.stack([torch.min(self.targets.critic(x, a), axis=1) for a in advice])
        index = torch.argmax(q, dim=0)
        advice = torch.stack(advice)
        best_advice = advice[index.squeeze(-1), range(advice.shape[1])]
        return best_advice, index

