from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import sys
import os
import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3.common import logger
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import get_linear_fn, polyak_update
from stable_baselines3.dqn.policies import DQNPolicy
from itertools import chain

sys.path.append(os.path.join(sys.path[0], 'src/icmppo'))
from src.icmppo.envs import Converter
from src.rewardprediction import RewardPredictorFactory
from src.icmppo.normalizers import StandardNormalizer, Normalizer, NoNormalizer


class DCQN(OffPolicyAlgorithm):
    """
    Deep Curiosity Q-Network (DCQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Set to `-1` to disable.
    :param gradient_steps: How many gradient steps to do after each rollout
        (see ``train_freq`` and ``n_episodes_rollout``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param n_episodes_rollout: Update the model every ``n_episodes_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``. Set to `-1` to disable.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
            self,
            policy: Union[str, Type[DQNPolicy]],
            env: Union[GymEnv, str],
            curiosity_factory,
            reward_predictor_factory: RewardPredictorFactory,
            learning_rate: Union[float, Callable] = 1e-4,
            buffer_size: int = 1000000,
            learning_starts: int = 50000,
            batch_size: Optional[int] = 32,
            tau: float = 1.0,
            gamma: float = 0.99,
            train_freq: int = 4,
            gradient_steps: int = 1,
            n_episodes_rollout: int = -1,
            optimize_memory_usage: bool = False,
            target_update_interval: int = 10000,
            exploration_fraction: float = 0.1,
            exploration_initial_eps: float = 1.0,
            exploration_final_eps: float = 0.05,
            max_grad_norm: float = 10,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            steps_per_log=0,
            normalize_state: bool = False,
            normalize_reward: bool = False
    ):

        super(DCQN, self).__init__(
            policy,
            env,
            DQNPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            n_episodes_rollout,
            action_noise=None,  # No action noise
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self.q_net, self.q_net_target = None, None

        if _init_setup_model:
            self._setup_model()

        self.state_converter = Converter.for_space(self.env.observation_space)
        self.action_converter = Converter.for_space(self.env.action_space)
        self.curiosity = curiosity_factory.create(self.state_converter, self.action_converter)
        self.curiosity.to(self.device, th.float)
        self.reward_predictor = reward_predictor_factory.create(self.state_converter, self.action_converter)
        self.reward_predictor.to(self.device, th.float)
        self.policy.optimizer = self.policy.optimizer_class(
            chain(self.policy.parameters(), self.curiosity.parameters(), self.reward_predictor.parameters()),
            lr=learning_rate, **self.policy.optimizer_kwargs)

        # initialize for hparams metric
        self.extrinsic_reward_sum = 0.0
        self.reward_normalizer = StandardNormalizer() if normalize_reward else NoNormalizer()
        self.state_normalizer = self.state_converter.state_normalizer() if normalize_state else NoNormalizer()

    def _setup_model(self) -> None:
        super(DCQN, self)._setup_model()
        self._create_aliases()
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps, self.exploration_final_eps, self.exploration_fraction
        )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollout()`` after each step in the environment.
        """
        if self.num_timesteps % self.target_update_interval == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        logger.record("rollout/exploration rate", self.exploration_rate)

    def model_rewards(self, batch_rewards, batch_states, batch_actions, batch_next_states):
        # calculate icm rewards, rp rewards, overall rewards in batch
        action_shape = self.env.action_space.shape
        # if action space shape is empty use shape (1) instead
        if action_shape == th.tensor(0).shape:
            action_shape = (1,)
        batch_actions = batch_actions.reshape((batch_actions.shape[0],) + action_shape)
        batch_rewards = batch_rewards.reshape(-1, 1)
        last_states = th.cat([batch_states[:1], batch_next_states], dim=0)

        with th.no_grad():
            batch_rewards = batch_rewards.cpu().numpy()
            batch_actions = batch_actions.cpu().numpy()
            batch_states = batch_states.cpu().numpy()
            last_states = last_states.cpu().numpy()

            icm_combined_rewards, icm_intrinsic_rewards, extrinsic_rewards, pred = \
                self.curiosity.reward(batch_rewards, last_states, batch_actions)
            rm_combined_rewards, rm_intrinsic_rewards, _, rm_predicted_rewards = \
                self.reward_predictor.reward(batch_rewards, batch_states, batch_actions)
            rewards = self.curiosity.intrinsic_reward_integration * icm_intrinsic_rewards + self.reward_predictor.intrinsic_reward_integration * rm_intrinsic_rewards + (
                        1.0 - self.curiosity.intrinsic_reward_integration - self.reward_predictor.intrinsic_reward_integration) * extrinsic_rewards
        rewards = th.tensor(rewards).to(self.device)
        return rewards

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        value_losses = []
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            batch_states = replay_data.observations.float()
            batch_actions = replay_data.actions.float().squeeze()
            batch_next_states = replay_data.next_observations.float()
            batch_extrinsic_rewards = replay_data.rewards.float()

            with th.no_grad():
                # calculate icm rewards, rp rewards, overall rewards in batch
                combined_rewards = self.model_rewards(batch_extrinsic_rewards, batch_states, batch_actions,
                                                      batch_next_states)
                # reward normalize
                norm_combined_rewards = self.reward_normalizer.partial_fit_transform(combined_rewards)
                # Compute the target Q values
                target_q = self.q_net_target(batch_next_states)
                # Follow greedy policy: use the one with the highest value
                target_q, _ = target_q.max(dim=1)
                # Avoid potential broadcast issue
                target_q = target_q.reshape(-1, 1)
                # 1-step TD target
                target_q = norm_combined_rewards + (1 - replay_data.dones) * self.gamma * target_q

            # Get current Q estimates
            current_q = self.q_net(batch_states)

            # Retrieve the q-values for the actions from the replay buffer
            current_q = th.gather(current_q, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q, target_q)
            value_losses.append(loss.item())

            loss, icm_policy_loss, icm_forward_loss, icm_inverse_loss, icm_curiosity_loss = \
                self.curiosity.loss(loss, batch_states, batch_next_states, batch_actions)
            rp_loss = self.reward_predictor.loss(batch_states, batch_actions, batch_extrinsic_rewards)
            loss += rp_loss

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/value_loss", np.mean(value_losses))

        logger.record("icm/policy_loss", icm_policy_loss.item())
        logger.record("icm/forward_loss", icm_forward_loss.item())
        logger.record("icm/inverse_loss", icm_inverse_loss.item())
        logger.record("icm/curiosity_loss", icm_curiosity_loss.item())
        logger.record("icm/replay_extrinsic_reward_mean", batch_extrinsic_rewards.mean().item())
        logger.record("icm/replay_combined_reward_mean", norm_combined_rewards.mean().item())

        logger.record("RewardPredictor/loss", rp_loss.item())

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            n_batch = observation.shape[0]
            action = np.array([self.action_space.sample() for _ in range(n_batch)])
        else:
            action, state = self.policy.predict(observation, state, mask, deterministic)
        return action, state

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "DCQN",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(DCQN, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(DCQN, self)._excluded_save_params() + ["q_net", "q_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
