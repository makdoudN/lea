# Residual Policy Learning.
import torch
import random
import numpy as np

from .ddpg import DDPG, TD3


class RPL(DDPG):
    def __init__(self, base_policy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_policy = base_policy

    @torch.no_grad()
    def act(self, obses: torch.Tensor, mode: str = "train"):
        action = self.get_action(obses).cpu().numpy()
        if not mode == "train":
            return np.clip(action, -1, 1)
        action += self.action_epsilon_nz * np.random.randn(*action.shape)
        action = np.clip(action, -1, 1)
        action += np.random.binomial(1, self.action_epsilon_rnd, 1)[0] * (
            np.random.uniform(low=-1, high=1, size=action.shape) - action
        )
        return action, {}

    def learn_critic(self, obses, reward, action, mask, next_obses):
        with torch.no_grad():
            next_action = self.get_action(next_obses, use_target=True)
            next_qfunc = self.targets.critic(next_obses, next_action)
            target = reward + self.gamma * mask * next_qfunc.squeeze(1)
        qfunc = self.models.critic(obses, action).squeeze(1)
        assert target.shape == qfunc.shape
        loss = (qfunc - target).pow(2).mean().mul(0.5)
        if self.logger is not None:
            self.logger.store(loss_critic=np.round(loss.item(), 3))


    def learn_actor(self, obses):
        action = self.get_action(obses)
        action_penalty = self.action_penalty_lam * action.pow(2).mean()
        loss = self.models.critic(obses, action).mean().neg()
        loss += action_penalty
        if self.logger is not None:
            self.logger.store(loss_actor=np.round(loss.item(), 3))
        return loss

    def get_action(self, context, use_target: bool = False):
        model = self.models.actor if not use_target else self.targets.actor
        return torch.clamp(model(context) + self.base_policy(context), -1, 1)


class RPLTD3(TD3):
    def __init__(self, base_policy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_policy = base_policy

    @torch.no_grad()
    def act(self, obses: torch.Tensor, mode: str = "train"):
        action = self.get_action(obses).cpu().numpy()
        if not mode == "train":
            return action
        action += self.action_epsilon_nz * np.random.randn(*action.shape)
        action = np.clip(action, -1, 1)
        action += np.random.binomial(1, self.action_epsilon_rnd, 1)[0] * (
            np.random.uniform(low=-1, high=1, size=action.shape) - action
        )
        return action, {}

    def learn_critic(self, obses, reward, action, mask, next_obses):
        with torch.no_grad():
            next_action = self.get_action(next_obses, use_target=True)
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
        return loss

    def learn_actor(self, obses):
        action = self.get_action(obses)
        action_penalty = self.action_penalty_lam * action.pow(2).mean()
        loss = self.models.critic(obses, action)[0].mean().neg()
        loss += action_penalty
        return loss

    def get_action(self, context, use_target: bool = False):
        model = self.models.actor if not use_target else self.targets.actor
        return torch.clamp(model(context) + self.base_policy(context), -1, 1)
