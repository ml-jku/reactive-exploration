from abc import ABCMeta, abstractmethod
from os import stat
from typing import Generator

import numpy as np
import torch
from torch import Tensor, nn
from torch._C import set_flush_denormal
from stable_baselines3.common.running_mean_std import RunningMeanStd

from curiosity.base import Curiosity, CuriosityFactory
from envs import Converter
# from icmppo.rewards import reward
from reporters import Reporter, NoReporter


class ICMModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, state_converter: Converter, action_converter: Converter):
        super().__init__()
        self.state_converter = state_converter
        self.action_converter = action_converter

    @property
    @abstractmethod
    def recurrent(self) -> bool:
        raise NotImplementedError('Implement me')

    @staticmethod
    @abstractmethod
    def factory() -> 'ICMModelFactory':
        raise NotImplementedError('Implement me')


class ICMModelFactory:
    def create(self, state_converter: Converter, action_converter: Converter) -> ICMModel:
        raise NotImplementedError('Implement me')


class ForwardModel(nn.Module):
    def __init__(self, action_converter: Converter, state_latent_features: int):
        super().__init__()
        self.action_converter = action_converter
        action_latent_features = 4
        if action_converter.discrete:
            self.action_encoder = nn.Embedding(action_converter.shape[0], action_latent_features)
        else:
            self.action_encoder = nn.Linear(action_converter.shape[0], action_latent_features)
        self.hidden = nn.Sequential(
            nn.Linear(action_latent_features + state_latent_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, state_latent_features)
        )

    def forward(self, state_latent: Tensor, action: Tensor):
        action = self.action_encoder(action.long() if self.action_converter.discrete else action).squeeze()
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        x = torch.cat((action, state_latent), dim=-1)
        x = self.hidden(x)
        return x

class MlpReconstructForwardModel(nn.Module):
    def __init__(self, action_converter: Converter, state_latent_features: int):
        super().__init__()
        self.action_converter = action_converter
        action_latent_features = 4
        if action_converter.discrete:
            self.action_encoder = nn.Embedding(action_converter.shape[0], action_latent_features)
        else:
            self.action_encoder = nn.Linear(action_converter.shape[0], action_latent_features)
        self.hidden = nn.Sequential(
            nn.Linear(action_latent_features + state_latent_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, state_latent_features)
        )

    def forward(self, state_latent: Tensor, action: Tensor):
        action = self.action_encoder(action.long() if self.action_converter.discrete else action).squeeze()
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        x = torch.cat((action, state_latent), dim=-1)
        x = self.hidden(x)
        return x


class InverseModel(nn.Module):
    def __init__(self, action_converter: Converter, state_latent_features: int):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(state_latent_features * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            action_converter.policy_out_model(128)
        )

    def forward(self, state_latent: Tensor, next_state_latent: Tensor):
        return self.input(torch.cat((state_latent, next_state_latent), dim=-1))


class ReconstructForwardModel(nn.Module):
    """
    class for ICMModel that predicts the 3d obs input space
    creates action embedding of shape W*H, concatenates it to the input state as an additional channel, and creates a prediction with the same shape as input observations
    """
    def __init__(self, action_converter: Converter, state_converter: Converter):
        super().__init__()
        self.action_converter = action_converter
        self.state_converter = state_converter
        # embedding of same shape as 2d state 1 channel
        action_latent_features = state_converter.shape[1] * state_converter.shape[2]
        n_input_channels = state_converter.shape[0]
        if action_converter.discrete:
            self.action_encoder = nn.Embedding(action_converter.shape[0], action_latent_features)
        else:
            self.action_encoder = nn.Linear(action_converter.shape[0], action_latent_features)

        # conv layers that take 2 * obs channels (because we will stack action encoder and state in forward)
        # and to produce the observation space shape as output (similar to reconstructICMModel zfnet style)
        self.hidden = nn.Sequential(
            nn.Conv2d(n_input_channels + 1, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, n_input_channels, kernel_size=3, stride=2, padding=0),
        )
        
        # code for repeating copies of action as an additional colour channel

        # super().__init__()
        # self.action_converter = action_converter
        # self.state_converter = state_converter
        # # embedding of same shape as 2d state 1 channel
        # n_input_channels = state_converter.shape[0]
        
        # # conv layers need an additional obs channel (because we will stack action encoder and state in forward)
        # # and to produce the observation space shape as output (similar to reconstructICMModel zfnet style)
        # self.hidden = nn.Sequential(
        #     nn.Conv2d(n_input_channels + 1, 32, kernel_size=3, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, n_input_channels, kernel_size=3, stride=2, padding=0),
        # )
        

    def forward(self, state_latent: Tensor, action: Tensor):
        action = self.action_encoder(action.long() if self.action_converter.discrete else action).squeeze()
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        # reshape action after encoder to (-1, 1, 11, 11) dynamically to obs space shape -1, 1, H, W
        action = action.reshape(-1, 1, self.state_converter.shape[1], self.state_converter.shape[2])
        # stack encoded action and state along channels before giving to hidden
        x = torch.cat((action, state_latent), dim=1)
        x = self.hidden(x)
        return x
        
        # code for repeating copies of action as an additional colour channel

        # # repeat each scalar action and turn them into 1xHxW duplicates of the original
        # action = action.repeat(self.state_converter.shape[1]*self.state_converter.shape[2], 1)
        # action = action.T
        # # reshape action to (-1, 1, 11, 11) dynamically to obs space shape -1, 1, H, W
        # action = action.reshape(-1, 1, self.state_converter.shape[1], self.state_converter.shape[2])
        # # stack encoded action and state along channels before giving to hidden
        # x = torch.cat((action, state_latent), dim=1)
        # x = self.hidden(x)
        # return x

class ReconstructInverseModel(nn.Module):
    # 2d state and 2d next_state are combined by concatenating their channels
    def __init__(self, action_converter: Converter, state_converter: Converter):
        super().__init__()
        n_input_channels = state_converter.shape[0]
        # reuse implementation from ReconstructedForwardModel here that takes 2*C, H, W, as the 2 states are being stacked
        self.cnn = nn.Sequential(
            nn.Conv2d(2*n_input_channels, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.rand(size= (1,) + (state_converter.shape[0]*2, state_converter.shape[1], state_converter.shape[2]) ).float()
            ).shape[1]
        # plug in correct shapes for linear layer
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU(),
            action_converter.policy_out_model(128)
        )
        self.input = nn.Sequential(self.cnn, self.linear)

    def forward(self, state_latent: Tensor, next_state_latent: Tensor):
        # stack states along channels before handing to self.input
        # TODO debug whether dim=1 is channels, I think 0 would be batch
        return self.input(torch.cat((state_latent, next_state_latent), dim=1))


class CNNStateReconstructModel(ICMModel):
    def __init__(self, state_converter: Converter, action_converter: Converter):
        super().__init__(state_converter, action_converter)
        # n_input_channels = state_converter.shape[0]

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, n_input_channels, kernel_size=3, stride=2, padding=0),
        # )

        self.forward_model = ReconstructForwardModel(action_converter, state_converter)
        # self.inverse_model = ReconstructInverseModel(action_converter, state_converter)

    @property
    def recurrent(self) -> bool:
        # to enable batch-wise reward prediction
        return True

    def forward(self, state: Tensor, next_state: Tensor, action: Tensor):
        # state = self.encoder(state)
        # next_state = self.encoder(next_state)
        next_state_hat = self.forward_model(state, action)
        # action_hat = self.inverse_model(state, next_state)

        # just returning nex_state and action because this should follow the same signature as icm even though no encoder is used
        return next_state, next_state_hat, action

    @staticmethod
    def factory() -> 'ICMModelFactory':
        return CNNStateReconstructModelFactory()


class CNNStateReconstructModelFactory(ICMModelFactory):
    def create(self, state_converter: Converter, action_converter: Converter) -> ICMModel:
        return CNNStateReconstructModel(state_converter, action_converter)


class CNNICMReconstructModel(ICMModel):
    def __init__(self, state_converter: Converter, action_converter: Converter):
        super().__init__(state_converter, action_converter)
        n_input_channels = state_converter.shape[0]

        self.encoder = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, n_input_channels, kernel_size=3, stride=2, padding=0),
        )

        self.forward_model = ReconstructForwardModel(action_converter, state_converter)
        self.inverse_model = ReconstructInverseModel(action_converter, state_converter)

    @property
    def recurrent(self) -> bool:
        # to enable batch-wise reward prediction
        return True

    def forward(self, state: Tensor, next_state: Tensor, action: Tensor):
        state = self.encoder(state)
        next_state = self.encoder(next_state)
        next_state_hat = self.forward_model(state, action)
        action_hat = self.inverse_model(state, next_state)
        return next_state, next_state_hat, action_hat

    @staticmethod
    def factory() -> 'ICMModelFactory':
        return CNNICMReconstructModelFactory()


class CNNICMReconstructModelFactory(ICMModelFactory):
    def create(self, state_converter: Converter, action_converter: Converter) -> ICMModel:
        return CNNICMReconstructModel(state_converter, action_converter)

class CNNICMModel(ICMModel):
    def __init__(self, state_converter: Converter, action_converter: Converter):
        super().__init__(state_converter, action_converter)
        n_input_channels = state_converter.shape[0]

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
                torch.rand(size= (1,) + state_converter.shape ).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, 128), nn.ReLU())
        self.encoder = nn.Sequential(self.cnn, self.linear)

        self.forward_model = ForwardModel(action_converter, 128)
        self.inverse_model = InverseModel(action_converter, 128)

    @property
    def recurrent(self) -> bool:
        # to enable batch-wise reward prediction
        return True

    def forward(self, state: Tensor, next_state: Tensor, action: Tensor):
        state = self.encoder(state)
        next_state = self.encoder(next_state)
        next_state_hat = self.forward_model(state, action)
        action_hat = self.inverse_model(state, next_state)
        return next_state, next_state_hat, action_hat

    @staticmethod
    def factory() -> 'ICMModelFactory':
        return CNNICMModelFactory()


class CNNICMModelFactory(ICMModelFactory):
    def create(self, state_converter: Converter, action_converter: Converter) -> ICMModel:
        return CNNICMModel(state_converter, action_converter)


class MlpICMModel(ICMModel):
    def __init__(self, state_converter: Converter, action_converter: Converter):
        assert len(state_converter.shape) == 1, 'Only flat spaces supported by MLP model'
        assert len(action_converter.shape) == 1, 'Only flat action spaces supported by MLP model'
        super().__init__(state_converter, action_converter)
        self.encoder = nn.Sequential(
            nn.Linear(state_converter.shape[0], 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),

        )
        self.forward_model = ForwardModel(action_converter, 128)
        self.inverse_model = InverseModel(action_converter, 128)

    @property
    def recurrent(self) -> bool:
        # enable batch-wise
        return True

    def forward(self, state: Tensor, next_state: Tensor, action: Tensor):
        state = self.encoder(state)
        next_state = self.encoder(next_state)
        next_state_hat = self.forward_model(state, action)
        action_hat = self.inverse_model(state, next_state)
        return next_state, next_state_hat, action_hat

    @staticmethod
    def factory() -> 'ICMModelFactory':
        return MlpICMModelFactory()


class MlpStateReconstructModel(ICMModel):
    def __init__(self, state_converter: Converter, action_converter: Converter):
        assert len(state_converter.shape) == 1, 'Only flat spaces supported by MLP model'
        assert len(action_converter.shape) == 1, 'Only flat action spaces supported by MLP model'
        super().__init__(state_converter, action_converter)
        # self.encoder = nn.Sequential(
        #     nn.Linear(state_converter.shape[0], 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 4),

        # )
        self.forward_model = MlpReconstructForwardModel(action_converter, 4)
        # self.inverse_model = InverseModel(action_converter, 128)

    @property
    def recurrent(self) -> bool:
        # enable batch-wise
        return True

    def forward(self, state: Tensor, next_state: Tensor, action: Tensor):
        # state = self.encoder(state)
        # next_state = self.encoder(next_state)
        next_state_hat = self.forward_model(state, action)
        # action_hat = self.inverse_model(state, next_state)
        return next_state, next_state_hat, action

    @staticmethod
    def factory() -> 'ICMModelFactory':
        return MlpStateReconstructModelFactory()

class MlpStateReconstructModelFactory(ICMModelFactory):
    def create(self, state_converter: Converter, action_converter: Converter) -> ICMModel:
        return MlpStateReconstructModel(state_converter, action_converter)

class MlpICMModelFactory(ICMModelFactory):
    def create(self, state_converter: Converter, action_converter: Converter) -> ICMModel:
        return MlpICMModel(state_converter, action_converter)


class StateModel(Curiosity):
    """
    Implements the Intrinsic Curiosity Module described in paper: https://arxiv.org/pdf/1705.05363.pdf

    The overview of the idea is to reward the agent for exploring unseen states. It is achieved by implementing two
    models. One called forward model that given the encoded state and encoded action computes predicts the encoded next
    state. The other one called inverse model that given the encoded state and encoded next_state predicts action that
    must have been taken to move from one state to the other. The final intrinsic reward is the difference between
    encoded next state and encoded next state predicted by the forward module. Inverse model is there to make sure agent
    focuses on the states that he actually can control.
    """

    def __init__(self, state_converter: Converter, action_converter: Converter, model_factory: ICMModelFactory,
                 policy_weight: float, reward_scale: float, weight: float, intrinsic_reward_integration: float,
                 reporter: Reporter):
        """

        :param state_converter: state converter
        :param action_converter: action converter
        :param model_factory: model factory
        :param policy_weight: weight to be applied to the ``policy_loss`` in the ``loss`` method. Allows to control how
               important optimizing policy to optimizing the curiosity module
        :param reward_scale: scales the intrinsic reward returned by this module. Can be used to control how big the
               intrinsic reward is
        :param weight: balances the importance between forward and inverse model
        :param intrinsic_reward_integration: balances the importance between extrinsic and intrinsic reward. Used when
               incorporating intrinsic into extrinsic in the ``reward`` method
        :param reporter reporter used to report training statistics
        """
        super().__init__(state_converter, action_converter)
        self.model: ICMModel = model_factory.create(state_converter, action_converter)
        self.policy_weight = policy_weight
        self.reward_scale = reward_scale
        self.weight = weight
        self.intrinsic_reward_integration = intrinsic_reward_integration
        self.reporter = reporter

    def parameters(self) -> Generator[nn.Parameter, None, None]:
        return self.model.parameters()

    def predict_next_state(self, state: Tensor, action: Tensor):
        with torch.no_grad():
            next_state_hat = self.model.forward_model(state, action)
        return next_state_hat

    def reward(self, rewards: np.ndarray, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        n, t = actions.shape[0], actions.shape[1]
        states, next_states = states[:-1], states[1:]
        states, next_states, actions = self._to_tensors(
            self.state_converter.reshape_as_input(states, self.model.recurrent),
            self.state_converter.reshape_as_input(next_states, self.model.recurrent),
            actions.reshape(n * t, *actions.shape[2:]))
        states = states.float()
        next_states = next_states.float()
        next_states_latent, next_states_hat, _ = self.model(states, next_states, actions)
        # l2 norm across all but first dimension
        intrinsic_reward = self.reward_scale / 2 * (next_states_hat - next_states_latent).norm(2, dim=tuple(range(1, next_states_hat.ndim))).pow(2)
        intrinsic_reward = intrinsic_reward.cpu().detach().numpy().reshape(n, t)

        combined_reward = (1. - self.intrinsic_reward_integration) * rewards + self.intrinsic_reward_integration * intrinsic_reward
        return combined_reward, intrinsic_reward, rewards, next_states_hat

    def loss(self, policy_loss: Tensor, states: Tensor, next_states: Tensor, actions: Tensor) -> Tensor:
        next_states_latent, next_states_hat, actions_hat = self.model(states, next_states, actions)
        curiosity_loss = forward_loss = 0.5 * (next_states_hat - next_states_latent.detach()).norm(2, dim=-1).pow(2).mean()
        # inverse_loss = self.action_converter.distance(actions_hat, actions)
        # curiosity_loss = self.weight * forward_loss + (1 - self.weight) * inverse_loss
        return self.policy_weight * policy_loss + curiosity_loss, policy_loss, forward_loss, torch.tensor(0.0), curiosity_loss

    def to(self, device: torch.device, dtype: torch.dtype):
        super().to(device, dtype)
        self.model.to(device, dtype)

    @staticmethod
    def factory(model_factory: ICMModelFactory, policy_weight: float, reward_scale: float,
                weight: float, intrinsic_reward_integration: float, reporter: Reporter = NoReporter()) -> 'ICMFactory':
        """
        Creates the factory for the ``ICM``
        :param model_factory: model factory
        :param policy_weight: weight to be applied to the ``policy_loss`` in the ``loss`` method. Allows to control how
               important optimizing policy to optimizing the curiosity module
        :param reward_scale: scales the intrinsic reward returned by this module. Can be used to control how big the
               intrinsic reward is
        :param weight: balances the importance between forward and inverse model
        :param intrinsic_reward_integration: balances the importance between extrinsic and intrinsic reward. Used when
               incorporating intrinsic into extrinsic in the ``reward`` method
        :param reporter reporter used to report training statistics
        :return: factory
        """
        return StateModelFactory(model_factory, policy_weight, reward_scale, weight, intrinsic_reward_integration, reporter)

class StateModelFactory(CuriosityFactory):
    def __init__(self, model_factory: ICMModelFactory, policy_weight: float, reward_scale: float, weight: float,
                 intrinsic_reward_integration: float, reporter: Reporter):
        self.policy_weight = policy_weight
        self.model_factory = model_factory
        self.scale = reward_scale
        self.weight = weight
        self.intrinsic_reward_integration = intrinsic_reward_integration
        self.reporter = reporter

    def create(self, state_converter: Converter, action_converter: Converter):
        return StateModel(state_converter, action_converter, self.model_factory, self.policy_weight, self.scale, self.weight,
                   self.intrinsic_reward_integration, self.reporter)


class ICM(Curiosity):
    """
    Implements the Intrinsic Curiosity Module described in paper: https://arxiv.org/pdf/1705.05363.pdf

    The overview of the idea is to reward the agent for exploring unseen states. It is achieved by implementing two
    models. One called forward model that given the encoded state and encoded action computes predicts the encoded next
    state. The other one called inverse model that given the encoded state and encoded next_state predicts action that
    must have been taken to move from one state to the other. The final intrinsic reward is the difference between
    encoded next state and encoded next state predicted by the forward module. Inverse model is there to make sure agent
    focuses on the states that he actually can control.
    """

    def __init__(self, state_converter: Converter, action_converter: Converter, model_factory: ICMModelFactory,
                 policy_weight: float, reward_scale: float, weight: float, intrinsic_reward_integration: float,
                 reporter: Reporter, norm_rewards=False):
        """

        :param state_converter: state converter
        :param action_converter: action converter
        :param model_factory: model factory
        :param policy_weight: weight to be applied to the ``policy_loss`` in the ``loss`` method. Allows to control how
               important optimizing policy to optimizing the curiosity module
        :param reward_scale: scales the intrinsic reward returned by this module. Can be used to control how big the
               intrinsic reward is
        :param weight: balances the importance between forward and inverse model
        :param intrinsic_reward_integration: balances the importance between extrinsic and intrinsic reward. Used when
               incorporating intrinsic into extrinsic in the ``reward`` method
        :param reporter reporter used to report training statistics
        """
        super().__init__(state_converter, action_converter)
        self.model: ICMModel = model_factory.create(state_converter, action_converter)
        self.policy_weight = policy_weight
        self.reward_scale = reward_scale
        self.weight = weight
        self.intrinsic_reward_integration = intrinsic_reward_integration
        self.reporter = reporter
        self.norm_rewards = norm_rewards
        self.return_rms = RunningMeanStd(shape=())
        self.epsilon = 1e-8

    def parameters(self) -> Generator[nn.Parameter, None, None]:
        return self.model.parameters()

    def predict_next_state(self, state: Tensor, action: Tensor):
        with torch.no_grad():
            state = self.model.encoder(state)
            next_state_hat = self.model.forward_model(state, action)
        return next_state_hat

    def reward(self, rewards: np.ndarray, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        n, t = actions.shape[0], actions.shape[1]
        states, next_states = states[:-1], states[1:]
        states, next_states, actions = self._to_tensors(
            self.state_converter.reshape_as_input(states, self.model.recurrent),
            self.state_converter.reshape_as_input(next_states, self.model.recurrent),
            actions.reshape(n * t, *actions.shape[2:]))
        states = states.float()
        next_states = next_states.float()
        next_states_latent, next_states_hat, _ = self.model(states, next_states, actions)
        # l2 norm across all but first dimension
        intrinsic_reward = self.reward_scale / 2 * (next_states_hat - next_states_latent).norm(2, dim=tuple(range(1, next_states_hat.ndim))).pow(2)
        intrinsic_reward = intrinsic_reward.cpu().detach().numpy()
        if self.norm_rewards:
            self.return_rms.update(intrinsic_reward)
            intrinsic_reward = intrinsic_reward / np.sqrt(self.return_rms.var + self.epsilon)
        intrinsic_reward = intrinsic_reward.reshape(n, t)

        combined_reward = (1. - self.intrinsic_reward_integration) * rewards + self.intrinsic_reward_integration * intrinsic_reward
        return combined_reward, intrinsic_reward, rewards, next_states_hat

    def loss(self, policy_loss: Tensor, states: Tensor, next_states: Tensor, actions: Tensor) -> Tensor:
        next_states_latent, next_states_hat, actions_hat = self.model(states, next_states, actions)
        forward_loss = 0.5 * (next_states_hat - next_states_latent.detach()).norm(2, dim=-1).pow(2).mean()
        inverse_loss = self.action_converter.distance(actions_hat, actions)
        curiosity_loss = self.weight * forward_loss + (1 - self.weight) * inverse_loss
        return self.policy_weight * policy_loss + curiosity_loss, policy_loss, forward_loss, inverse_loss, curiosity_loss

    def to(self, device: torch.device, dtype: torch.dtype):
        super().to(device, dtype)
        self.model.to(device, dtype)

    @staticmethod
    def factory(model_factory: ICMModelFactory, policy_weight: float, reward_scale: float,
                weight: float, intrinsic_reward_integration: float, reporter: Reporter = NoReporter(),
                norm_rewards=False) -> 'ICMFactory':
        """
        Creates the factory for the ``ICM``
        :param model_factory: model factory
        :param policy_weight: weight to be applied to the ``policy_loss`` in the ``loss`` method. Allows to control how
               important optimizing policy to optimizing the curiosity module
        :param reward_scale: scales the intrinsic reward returned by this module. Can be used to control how big the
               intrinsic reward is
        :param weight: balances the importance between forward and inverse model
        :param intrinsic_reward_integration: balances the importance between extrinsic and intrinsic reward. Used when
               incorporating intrinsic into extrinsic in the ``reward`` method
        :param reporter reporter used to report training statistics
        :return: factory
        """
        return ICMFactory(model_factory, policy_weight, reward_scale, weight, intrinsic_reward_integration,
                          reporter, norm_rewards)


class ICMFactory(CuriosityFactory):
    def __init__(self, model_factory: ICMModelFactory, policy_weight: float, reward_scale: float, weight: float,
                 intrinsic_reward_integration: float, reporter: Reporter, norm_rewards=False):
        self.policy_weight = policy_weight
        self.model_factory = model_factory
        self.scale = reward_scale
        self.weight = weight
        self.intrinsic_reward_integration = intrinsic_reward_integration
        self.reporter = reporter
        self.norm_rewards = norm_rewards

    def create(self, state_converter: Converter, action_converter: Converter):
        return ICM(state_converter, action_converter, self.model_factory, self.policy_weight, self.scale, self.weight,
                   self.intrinsic_reward_integration, self.reporter, self.norm_rewards)
