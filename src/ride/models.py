import torch
import numpy as np
from src.icmppo.curiosity.icm import ICM


class RIDE(ICM):

    def __init__(self, state_converter, action_converter, model_factory,
                 policy_weight, reward_scale, weight, intrinsic_reward_integration, reporter, count_norm=False,
                 norm_rewards=False):
        super().__init__(state_converter, action_converter, model_factory, policy_weight, reward_scale,
                         weight, intrinsic_reward_integration, reporter, norm_rewards=norm_rewards)
        self.count_norm = count_norm

    def reward(self, rewards, states, actions, state_counts):
        n, t = actions.shape[0], actions.shape[1]
        states, next_states = states[:-1], states[1:]
        states, next_states, actions = self._to_tensors(
            self.state_converter.reshape_as_input(states, self.model.recurrent),
            self.state_converter.reshape_as_input(next_states, self.model.recurrent),
            actions.reshape(n * t, *actions.shape[2:]))
        states = states.float()
        next_states = next_states.float()
        # compute RIDE rewards
        # don't need to call model.forward() here --> just model.encoder()
        states_latent, next_states_latent = self.model.encoder(states), self.model.encoder(next_states)

        # l2 norm across all but first dimension
        intrinsic_reward = self.reward_scale / 2 * (next_states_latent - states_latent).norm(2, dim=tuple(
            range(1, next_states_latent.ndim))).pow(2)
        if self.count_norm:
            state_counts = self._to_tensors(state_counts)[0].float()
            intrinsic_reward *= 1 / torch.sqrt(state_counts)

        intrinsic_reward = intrinsic_reward.cpu().detach().numpy()
        if self.norm_rewards:
            self.return_rms.update(intrinsic_reward)
            intrinsic_reward = intrinsic_reward / np.sqrt(self.return_rms.var + self.epsilon)
        intrinsic_reward = intrinsic_reward.reshape(n, t)

        combined_reward = (
                                      1. - self.intrinsic_reward_integration) * rewards + self.intrinsic_reward_integration * intrinsic_reward
        return combined_reward, intrinsic_reward, rewards, next_states_latent
