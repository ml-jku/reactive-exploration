import torch
from .models import RND


class NovelD(RND):

    def __init__(self, state_converter, action_converter, model_factory, policy_weight, reward_scale,
                 weight, intrinsic_reward_integration, reporter, alpha=0.5, count_norm=False, norm_rewards=False,
                 norm_obs=False):
        super().__init__(state_converter, action_converter, model_factory, policy_weight, reward_scale,
                         weight, intrinsic_reward_integration, reporter, norm_rewards=norm_rewards, norm_obs=norm_obs)
        self.alpha = alpha
        self.count_norm = count_norm

    def reward(self, rewards, states, actions, state_counts):
        n, t = actions.shape[0], actions.shape[1]
        if self.norm_obs:
            states = self.normalize_obs(states)
        states, next_states = states[:-1], states[1:]
        states, next_states, actions = self._to_tensors(
            self.state_converter.reshape_as_input(states, self.model.recurrent),
            self.state_converter.reshape_as_input(next_states, self.model.recurrent),
            actions.reshape(n * t, *actions.shape[2:]))
        states = states.float()
        next_states = next_states.float()
        # compute RND predictions --> novelty
        preds_current, targets_current = self.model(states)
        preds_next, targets_next = self.model(next_states)
        # compute current novelty, next novelty
        intrinsic_reward_current = (preds_current - targets_current).pow(2).sum(1) * (self.reward_scale / 2)
        intrinsic_reward_next = (preds_next - targets_next).pow(2).sum(1) * (self.reward_scale / 2)
        # compute intrinsic reward
        intrinsic_reward = torch.clamp(intrinsic_reward_next - self.alpha * intrinsic_reward_current, min=0)
        if self.count_norm:
            state_counts = self._to_tensors(state_counts)[0].float()
            intrinsic_reward *= (state_counts == 1).float()
        intrinsic_reward = intrinsic_reward.cpu().detach().numpy()
        if self.norm_rewards:
            intrinsic_reward = self.normalize_rewards(intrinsic_reward)
        intrinsic_reward = intrinsic_reward.reshape(n, t)

        combined_reward = (1. - self.intrinsic_reward_integration) * rewards \
                          + self.intrinsic_reward_integration * intrinsic_reward
        return combined_reward, intrinsic_reward, rewards, None
