"""
Random network distillation: https://arxiv.org/abs/1810.12894

RND implementation adjusted from:
    https://github.com/jcwleo/random-network-distillation-pytorch/blob/master/model.py

Follows the same structure as icmppo.

"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from stable_baselines3.common.running_mean_std import RunningMeanStd


class RNDModel(nn.Module):

    def __init__(self, state_converter, action_converter, latent_dim=512):
        super().__init__()
        self.state_converter = state_converter
        self.action_converter = action_converter
        n_input_channels = state_converter.shape[0]
        cnn = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=32, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy_input = torch.rand(
                size=(1,) + (state_converter.shape[0], state_converter.shape[1], state_converter.shape[2])).float()
            n_flatten = cnn(dummy_input).shape[1]

        linear = nn.Sequential(
            nn.Linear(n_flatten, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # make nets
        self.predictor = nn.Sequential(cnn, linear)
        self.target = copy.deepcopy(self.predictor)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        return self.predictor(next_obs), self.target(next_obs)

    @property
    def recurrent(self) -> bool:
        # to enable batch-wise reward prediction
        return True


class RNDMlpModel(nn.Module):

    def __init__(self, state_converter, action_converter, latent_dim=512):
        super().__init__()
        self.state_converter = state_converter
        self.action_converter = action_converter
        linear = nn.Sequential(
            nn.Linear(state_converter.shape[0], latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # make nets
        self.predictor = nn.Sequential(linear)
        self.target = copy.deepcopy(self.predictor)

        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        return self.predictor(next_obs), self.target(next_obs)

    @property
    def recurrent(self) -> bool:
        # to enable batch-wise reward prediction
        return True


class RND:

    def __init__(self, state_converter, action_converter, model_factory, policy_weight, reward_scale,
                 weight, intrinsic_reward_integration, reporter, norm_rewards=False, norm_obs=False):
        self.state_converter = state_converter
        self.action_converter = action_converter
        self.model = model_factory.create(state_converter, action_converter)
        print(self.model)
        self.policy_weight = policy_weight
        self.reward_scale = reward_scale
        self.weight = weight
        self.intrinsic_reward_integration = intrinsic_reward_integration
        self.reporter = reporter
        self.norm_rewards = norm_rewards
        self.norm_obs = norm_obs
        self.return_rms = RunningMeanStd(shape=())
        self.obs_rms = RunningMeanStd(shape=self.state_converter.shape)
        self.epsilon = 1e-8

    def reward(self, rewards, states, actions):
        n, t = actions.shape[0], actions.shape[1]
        if self.norm_obs:
            states = self.normalize_obs(states)

        next_states = states[1:]
        next_states, actions = self._to_tensors(
            self.state_converter.reshape_as_input(next_states, self.model.recurrent),
            actions.reshape(n * t, *actions.shape[2:])
        )
        next_states = next_states.float()
        preds, targets = self.model(next_states)
        intrinsic_reward = (preds - targets).pow(2).sum(1) * (self.reward_scale / 2)
        intrinsic_reward = intrinsic_reward.cpu().detach().numpy()
        if self.norm_rewards:
            intrinsic_reward = self.normalize_rewards(intrinsic_reward)

        intrinsic_reward = intrinsic_reward.reshape(n, t)
        combined_reward = (1. - self.intrinsic_reward_integration) * rewards \
                          + self.intrinsic_reward_integration * intrinsic_reward
        return combined_reward, intrinsic_reward, rewards, None

    def loss(self, policy_loss, states, next_states, actions):
        # redundant state, actions input, keep like this for now
        if self.norm_obs:
            next_states = self.normalize_obs(next_states, update_stats=False, to_tensor=True)
        preds, targets = self.model(next_states)
        rnd_loss = 0.5 * (preds - targets).norm(2, dim=-1).pow(2).mean()
        rnd_loss = rnd_loss * self.weight
        return self.policy_weight * policy_loss + rnd_loss, policy_loss, rnd_loss

    def parameters(self):
        return self.model.parameters()

    def to(self, device, dtype) -> None:
        self.device = device
        self.dtype = dtype
        self.model.to(device, dtype)

    def _to_tensors(self, *arrays: np.ndarray):
        return [torch.tensor(array, device=self.device, dtype=self.dtype) for array in arrays]

    def normalize_obs(self, states, update_stats=True, to_tensor=False):
        mean, std = self.obs_rms.mean, np.sqrt(self.obs_rms.var + self.epsilon)
        if to_tensor:
            mean = torch.from_numpy(mean).float().to(states.device)
            std = torch.from_numpy(std).float().to(states.device)
        if update_stats:
            self.obs_rms.update(states)
        return (states - mean) / std

    def normalize_rewards(self, r):
        self.return_rms.update(r)
        return r / np.sqrt(self.return_rms.var + self.epsilon)
