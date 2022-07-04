from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import sys
import os
import time
import collections
import gym
import numpy as np
import torch as th
import pandas as pd
from stable_baselines3.common import logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv

sys.path.append(os.path.join(sys.path[0], 'src/icmppo'))
from src.icmppo.normalizers import StandardNormalizer, Normalizer, NoNormalizer
from src.custom_buffers import RolloutBuffer
from src.rnd import NovelD
from src.ride import RIDE


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            curiosity_factory,
            reward_predictor_factory,
            learning_rate: Union[float, Callable],
            n_steps: int,
            gamma: float,
            gae_lambda: float,
            ent_coef: float,
            vf_coef: float,
            max_grad_norm: float,
            use_sde: bool,
            sde_sample_freq: int,
            normalize_state: bool = False,
            normalize_reward: bool = False,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            monitor_wrapper: bool = True,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            state_prediction_log_interval=None
    ):

        super(OnPolicyAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=ActorCriticPolicy,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        # initialize for hparams metric
        self.extrinsic_reward_sum = 0.0
        self.state_prediction_log_interval = state_prediction_log_interval
        self.reward_normalizer = StandardNormalizer() if normalize_reward else NoNormalizer()
        self.state_normalizer = self.state_converter.state_normalizer() if normalize_state else NoNormalizer()
        self.predicitons_df = pd.DataFrame()
        self.curiosity_factory = curiosity_factory
        self.reward_predictor_factory = reward_predictor_factory

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = RolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
            self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int
    ) -> bool:
        """
        Collect rollouts using the current policy and fill a `RolloutBuffer`.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        # buffers calculating reward/state model predictions and for logging
        self.icm_combined_rewards = np.zeros(n_rollout_steps)
        self.icm_intrinsic_rewards = np.zeros(n_rollout_steps)
        self.extrinsic_rewards = np.zeros(n_rollout_steps)

        # for RIDE and NovelD
        self.episode_state_count_dict = collections.defaultdict(int)
        self.episode_state_count = np.zeros(n_rollout_steps)

        self.rm_combined_rewards = np.zeros(n_rollout_steps)
        self.rm_intrinsic_rewards = np.zeros(n_rollout_steps)
        self.rm_predicted_rewards = np.zeros(n_rollout_steps)

        # obs buffer has to be 1 longer starting with initial reset observation and then predicting next obs / next state, e.g. for 2048 steps 2049 observations
        self._last_obs_buffer = np.zeros((n_rollout_steps + 1,) + env.observation_space.shape)
        self.actions_buffer = np.zeros((n_rollout_steps,) + env.action_space.shape)
        self.rewards_buffer = np.zeros(n_rollout_steps)
        self._last_dones_buffer = np.zeros(n_rollout_steps)
        self.values_buffer = th.zeros(n_rollout_steps)
        self.log_probs_buffer = th.zeros(n_rollout_steps)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            new_obs = self.state_normalizer.partial_fit_transform(new_obs)

            # rewards = self.reward_normalizer.partial_fit_transform(rewards)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # rollout_buffer.add(self._last_obs, actions, rewards, extrinsic_rewards, self._last_dones, values, log_probs)
            self._last_obs_buffer[n_steps - 1] = np.array(self._last_obs).copy()
            self.actions_buffer[n_steps - 1] = np.array(actions).copy()
            self.rewards_buffer[n_steps - 1] = np.array(rewards).copy()
            self._last_dones_buffer[n_steps - 1] = np.array(self._last_dones).copy()
            self.values_buffer[n_steps - 1] = values.clone()
            self.log_probs_buffer[n_steps - 1] = log_probs.clone()

            # for NovelD and RIDE
            obs_key = tuple(np.array(self._last_obs).copy().flatten())
            self.episode_state_count_dict[obs_key] += 1
            self.episode_state_count[n_steps - 1] = self.episode_state_count_dict.get(obs_key)

            self._last_obs = new_obs
            self._last_dones = dones

        # add the latest next_state from the very end of the rollout
        self._last_obs_buffer[n_steps] = new_obs

        # calculate icm rewards, rp rewards, overall rewards in batch
        rewards = self.model_rewards(env, n_rollout_steps)

        # reward normalize
        rewards = self.reward_normalizer.partial_fit_transform(rewards)
        self.rewards_buffer = rewards

        # add everything to rollout buffer
        for n_steps in range(len(self.rewards_buffer)):
            rollout_buffer.add(self._last_obs_buffer[n_steps], self.actions_buffer[n_steps],
                               self.rewards_buffer[n_steps], self.extrinsic_rewards[n_steps],
                               self._last_dones_buffer[n_steps], self.values_buffer[n_steps],
                               self.log_probs_buffer[n_steps])

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = th.as_tensor(new_obs).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def model_rewards(self, env, n_rollout_steps):
        # calculate icm rewards, rp rewards, overall rewards in batch
        action_shape = env.action_space.shape
        # if action space shape is empty use shape (1) instead
        if action_shape == th.tensor(0).shape:
            action_shape = (1,)
        self.actions_buffer = self.actions_buffer.reshape((n_rollout_steps,) + action_shape)
        self.rewards_buffer = self.rewards_buffer.reshape(-1, 1)

        with th.no_grad():
            if isinstance(self.curiosity, (NovelD, RIDE)):
                # additionally pass episode_state_counts reward()
                self.icm_combined_rewards, self.icm_intrinsic_rewards, self.extrinsic_rewards, pred = \
                    self.curiosity.reward(self.rewards_buffer, self._last_obs_buffer, self.actions_buffer,
                                          self.episode_state_count)
            else:
                self.icm_combined_rewards, self.icm_intrinsic_rewards, self.extrinsic_rewards, pred = \
                    self.curiosity.reward(self.rewards_buffer, self._last_obs_buffer, self.actions_buffer)
            if self.state_prediction_log_interval and self.state_prediction_log_interval[0] < self.num_timesteps < \
                    self.state_prediction_log_interval[1]:
                self.predicitons_df = self.predicitons_df.append(
                    pd.DataFrame(pred.reshape(self.batch_size - 1, -1).numpy()))
            self.rm_combined_rewards, self.rm_intrinsic_rewards, _, self.rm_predicted_rewards = \
                self.reward_predictor.reward(self.rewards_buffer, self._last_obs_buffer[:-1], self.actions_buffer)

            rewards = self.curiosity.intrinsic_reward_integration * self.icm_intrinsic_rewards + self.reward_predictor.intrinsic_reward_integration * self.rm_intrinsic_rewards + (
                        1.0 - self.curiosity.intrinsic_reward_integration - self.reward_predictor.intrinsic_reward_integration) * self.extrinsic_rewards
        return rewards

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "OnPolicyAlgorithm",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path,
            reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                logger.record("time/fps", fps)
                logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

                logger.record("icm/reward_intrinsic", self.icm_intrinsic_rewards.mean().item())
                logger.record("icm/reward_combined", self.icm_combined_rewards.mean().item())

                # with th.no_grad():
                #     self.rm_one_green_reward = self.reward_predictor.model(custom_observations.get_one_green_pickup_obs().to(self.device), th.tensor(0).to(self.device)).float().item()
                #     self.rm_one_red_reward = self.reward_predictor.model(custom_observations.get_one_red_pickup_obs().to(self.device), th.tensor(0).to(self.device)).float().item()
                #     self.rm_white_reward = self.reward_predictor.model(custom_observations.get_white_obs().to(self.device), th.tensor(0).to(self.device)).float().item()

                # logger.record("RewardPredictor/one_green_item_forward_reward", self.rm_one_green_reward)
                # logger.record("RewardPredictor/one_red_item_forward_reward", self.rm_one_red_reward)
                # logger.record("RewardPredictor/white_obs_forward_reward", self.rm_white_reward)

                logger.record("RewardPredictor/prediction_reward_combined", self.rm_combined_rewards.mean().item())
                logger.record("RewardPredictor/prediction_reward_intrinsic", self.rm_intrinsic_rewards.mean().item())
                logger.record("RewardPredictor/predicted_reward", self.rm_predicted_rewards.mean().item())

                logger.record("RewardPredictor+ICM/reward_rp+icm+env", self.rewards_buffer.mean().item())
                logger.record("RewardPredictor+ICM/reward_extrinsic", self.extrinsic_rewards.mean().item())

                self.extrinsic_reward_sum += self.extrinsic_rewards.sum().item()

                logger.dump(step=self.num_timesteps)

            self.train()

        self.predicitons_df.to_csv('test_logged_predictions.csv')
        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
