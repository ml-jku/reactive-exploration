from typing import Any, Callable, Dict, List, Optional, Type
import gym
from torch import nn
import torch as th
from torch.distributions import Categorical
from stable_baselines3.common.policies import BasePolicy, register_policy as common_register_policy
from stable_baselines3.dqn.policies import DQNPolicy, register_policy as dqn_register_policy

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    NatureCNN,
    create_mlp,
)


class SQLQNetwork(BasePolicy):
    """
    Action-Value (Q-Value) network for Soft Q-Learning.

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            features_extractor: nn.Module,
            features_dim: int,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
            alpha=0.1
    ):
        super(SQLQNetwork, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.alpha = alpha
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.normalize_images = normalize_images
        action_dim = self.action_space.n  # number of actions
        q_net = create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)
        self.q_net = nn.Sequential(*q_net)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        logits = self.q_net(self.extract_features(obs))
        return logits

    def get_value(self, q_value):
        q_ = th.clip(q_value, max=10)
        v_ = th.clip(th.sum(th.exp(q_ / self.alpha), dim=1, keepdim=True), min=1e-7)
        return self.alpha * th.log(v_)

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        with th.no_grad():
            q = self(observation)
            v = self.get_value(q).squeeze()
            d_ = th.clip((q - v) / self.alpha, min=-10, max=10)
            dist = th.exp(d_)
            c = Categorical(dist)
            a = c.sample()
        return a

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
                epsilon=self.epsilon,
            )
        )
        return data


class SQLCnnPolicy(DQNPolicy):
    """
    Policy class for Soft Q-Learning when using images as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(SQLCnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def make_q_net(self) -> SQLQNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return SQLQNetwork(**net_args).to(self.device)


common_register_policy("SQLCnnPolicy", SQLCnnPolicy)
dqn_register_policy("SQLCnnPolicy", SQLCnnPolicy)
