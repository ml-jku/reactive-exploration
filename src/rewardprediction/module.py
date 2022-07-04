from abc import ABCMeta
from typing import List, Generator

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import MSELoss
from stable_baselines3.common.running_mean_std import RunningMeanStd

from envs import Converter
from reporters import Reporter, NoReporter


class RewardModel(nn.Module):
    def __init__(self, state_converter: Converter, action_converter: Converter):
        super().__init__()
        self.action_converter = action_converter
        action_latent_features = 4
        n_input_channels = state_converter.shape[0]
        latent_dim = 128

        # cnn for state input
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.rand(size=(1,) + state_converter.shape).float()
            ).shape[1]

        # action encoder
        if action_converter.discrete:
            self.action_encoder = nn.Embedding(action_converter.shape[0], action_latent_features)
        else:
            self.action_encoder = nn.Linear(action_converter.shape[0], action_latent_features)

        # combine state encoding and action encoding, predict scalar reward
        self.hidden = nn.Sequential(
            nn.Linear(action_latent_features + n_flatten, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, 1)
        )

    def forward(self, state: Tensor, action: Tensor):
        action = self.action_encoder(action.long() if self.action_converter.discrete else action).squeeze()
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        state_latent = self.cnn(state)
        x = torch.cat((action, state_latent), dim=-1)
        x = self.hidden(x)
        return x

    @property
    def recurrent(self) -> bool:
        return False

    @staticmethod
    def factory() -> 'RewardModelFactory':
        return RewardModelFactory()


class RewardModelFactory(metaclass=ABCMeta):
    def create(self, state_converter: Converter, action_converter: Converter) -> RewardModel:
        return RewardModel(state_converter, action_converter)


class RewardPredictor(metaclass=ABCMeta):
    def __init__(self, state_converter: Converter, action_converter: Converter, model_factory: RewardModelFactory,
                 reporter: Reporter, loss_weight: float = 1.0, intrinsic_reward_scale: float = 1.0,
                 intrinsic_reward_integration: float = 0.0, norm_rewards=False):
        self.loss_weight = loss_weight
        self.intrinsic_reward_scale = intrinsic_reward_scale
        self.intrinsic_reward_integration = intrinsic_reward_integration
        self.reporter = reporter
        self.state_converter = state_converter
        self.action_converter = action_converter
        self.lossFunction = MSELoss()
        self.model = model_factory.create(state_converter=state_converter, action_converter=action_converter)
        self.norm_rewards = norm_rewards
        self.return_rms = RunningMeanStd(shape=())
        self.epsilon = 1e-8

    def parameters(self) -> Generator[nn.Parameter, None, None]:
        return self.model.parameters()

    def predict_reward(self, state: Tensor, action: Tensor):
        with torch.no_grad():
            return self.model.forward(state, action)

    def _to_tensors(self, *arrays: np.ndarray) -> List[torch.Tensor]:
        return [torch.tensor(array, device=self.device, dtype=self.dtype) for array in arrays]

    def reward(self, rewards: np.ndarray, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Not to be confused given the name "reward" - this method is used to generate an intrinsic reward for the agent based on how wrong the reward model's prediction is
        """
        rewards, states, actions = self._to_tensors(rewards, states, actions)
        rew_pred = self.model(states, actions)

        n, t = actions.shape[0], actions.shape[1]
        intrinsic_reward = self.intrinsic_reward_scale / 2 * (rew_pred - rewards).norm(2, dim=-1).pow(2)
        intrinsic_reward = intrinsic_reward.cpu().detach().numpy()

        if self.norm_rewards:
            self.return_rms.update(intrinsic_reward)
            intrinsic_reward = intrinsic_reward / np.sqrt(self.return_rms.var + self.epsilon)

        intrinsic_reward = intrinsic_reward.reshape(n, t)
        rewards = rewards.cpu().detach().numpy()

        combined_reward = (
                                      1. - self.intrinsic_reward_integration) * rewards + self.intrinsic_reward_integration * intrinsic_reward

        return combined_reward, intrinsic_reward, rewards, rew_pred

    def loss(self, states: Tensor, actions: Tensor, next_rewards: Tensor) -> Tensor:
        reward_hat = self.model(states, actions)
        loss = self.lossFunction(reward_hat, next_rewards) * self.loss_weight

        return loss

    @staticmethod
    def factory(model_factory: RewardModelFactory, reporter: Reporter = NoReporter(), loss_weight: float = 1.0,
                intrinsic_reward_scale: float = 1.0, intrinsic_reward_integration: float = 0.0,
                norm_rewards=False) -> 'RewardPredictorFactory':
        return RewardPredictorFactory(model_factory, reporter, loss_weight, intrinsic_reward_scale,
                                      intrinsic_reward_integration, norm_rewards=norm_rewards)

    def to(self, device: torch.device, dtype: torch.dtype):
        # super().to(device, dtype)
        self.device = device
        self.dtype = dtype
        self.model.to(device, dtype)


class RewardPredictorFactory(metaclass=ABCMeta):
    def __init__(self, model_factory: RewardModelFactory, reporter: Reporter, loss_weight: float = 1.0,
                 intrinsic_reward_scale: float = 1.0, intrinsic_reward_integration: float = 0.0, norm_rewards=False):
        self.model_factory = model_factory
        self.intrinsic_reward_scale = intrinsic_reward_scale
        self.intrinsic_reward_integration = intrinsic_reward_integration
        self.reporter = reporter
        self.loss_weight = loss_weight
        self.norm_rewards = norm_rewards

    def create(self, state_converter: Converter, action_converter: Converter):
        return RewardPredictor(state_converter, action_converter, self.model_factory,
                               self.reporter, self.loss_weight, self.intrinsic_reward_scale,
                               self.intrinsic_reward_integration, self.norm_rewards)


class MlpRewardModel(nn.Module):
    def __init__(self, state_converter: Converter, action_converter: Converter):
        assert len(state_converter.shape) == 1, 'Only flat spaces supported by MLP model'
        assert len(action_converter.shape) == 1, 'Only flat action spaces supported by MLP model'
        super().__init__()
        self.action_converter = action_converter
        action_latent_features = 2
        n_input_channels = state_converter.shape[0]

        # action encoder
        if action_converter.discrete:
            self.action_encoder = nn.Embedding(action_converter.shape[0], action_latent_features)
        else:
            self.action_encoder = nn.Linear(action_converter.shape[0], action_latent_features)

        # combine state encoding and action encoding, predict scalar reward
        self.encoder = nn.Sequential(
            nn.Linear(action_latent_features + state_converter.shape[0], 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, state: Tensor, action: Tensor):
        action = self.action_encoder(action.long() if self.action_converter.discrete else action).squeeze()
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        x = torch.cat((action, state), dim=-1)
        x = self.encoder(x)
        return x

    @property
    def recurrent(self) -> bool:
        return False

    @staticmethod
    def factory() -> 'RewardModelFactory':
        return MlpRewardModelFactory()


class MlpRewardModelFactory(metaclass=ABCMeta):
    def create(self, state_converter: Converter, action_converter: Converter) -> RewardModel:
        return MlpRewardModel(state_converter, action_converter)
