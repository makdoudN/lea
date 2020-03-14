# Learning from advice with auxiliary incentive.
from rltk.common.maths import polyak_averaging
from .ddpg import DDPG, TD3
import torch
import random
import numpy as np


class DDPG_LEA(DDPG):
    def __init__(
            self,
            library,
            pi_reuse,
            reuse: bool = True,
            num_commit: int = 1,
            lam_pi_guidance: float = 0.001,
            use_pi_guidance: bool = False,
            use_q_guidance: bool = False,
            lam_aux_policy_guidance: float = 0.001,
            aux_policy_guidance_version: int = 0,
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
        self.use_q_guidance = use_q_guidance
        self.use_pi_guidance = use_pi_guidance
        self.lam_pi_guidance = lam_pi_guidance
        self.count_advice = np.zeros(len(self.library))
        self.lam_aux_policy_guidance = lam_aux_policy_guidance
        self.aux_policy_guidance_version = aux_policy_guidance_version

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
            if not self.use_q_guidance:
                next_action = self.targets.actor(next_obses)
            else:
                advice = [advisor(next_obses) for advisor in self.library]
                next_action, _ = self.filter(next_obses, advice)
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
        q_sa = self.models.critic(obses, action)
        loss = q_sa.mean().neg()
        loss += action_penalty
        # Auxiliary Incentive to surpass Experts. --------------------------------------
        if self.use_q_guidance:
            advice = [advisor(obses) for advisor in self.library]
            if self.aux_policy_guidance_version == 0:
                q_sa = torch.stack([self.targets.critic(obses, a) for a in advice])
                q_sa_max = torch.max(q_sa, dim=0)[0]
                loss = loss + self.lam_aux_policy_guidance  * (q_sa - q_sa_max).pow(2).mean()
            elif self.lam_aux_policy_guidance == 1:
                raise # APPLY WITH SOFTMAX
                q_sa = torch.stack([self.targets.critic(obses, a) for a in advice])
                q_sa_max = torch.max(q_sa, dim=0)[0]
                loss = loss + self.lam_aux_policy_guidance  * (q_sa - q_sa_max).pow(2).mean()
            else:
                raise NotImplementedError()
        # -------------------------------------------------------------------------------
        if self.use_pi_guidance:
            advice = [advisor(obses) for advisor in self.library]
            best_advice, index = self.filter(obses, advice)
            loss_pi_guidance = self.lam_pi_guidance * \
                (action - best_advice).sum().pow(2).mul(0.5)
            assert loss.shape == loss_pi_guidance.shape
            loss = loss + loss_pi_guidance
            if self.logger is not None:
                self.logger.store(loss_pi_guidance=np.round(loss_pi_guidance.item(), 3))
        if self.logger is not None:
            self.logger.store(loss_actor=np.round(loss.item(), 3))
        return loss
