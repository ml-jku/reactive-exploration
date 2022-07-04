import math
import tensorflow as tf
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


class TBHparamCallback(BaseCallback):
    def __init__(self, hparam_dict, verbose=0):
        super(TBHparamCallback, self).__init__(verbose)
        self.hparam_dict = hparam_dict

    def on_training_end(self) -> None:
        try:
            output_formats = self.logger.Logger.CURRENT.output_formats
            self.tb_formatter = next(
                formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))
            w = self.tb_formatter.writer
            print('writing hparams to tb...')
            w.add_hparams(self.hparam_dict,
                          {'hparam/mean_extrinsic_reward': self.model.extrinsic_reward_sum / self.model.num_timesteps})
        except Exception:
            print(
                'Exception: No TensorBoardOutputFormat formatter found when trying to write hyperparameters in TBHparamCallback.on_training_end().')

    def _on_step(self):
        pass


class CartPoleTBHparamCallback(BaseCallback):
    def __init__(self, hparam_dict, env_config, verbose=0):
        super(CartPoleTBHparamCallback, self).__init__(verbose)
        self._merge(env_config, hparam_dict)
        self.hparam_dict = hparam_dict

    def _merge(self, dict1, dict2):
        return (dict2.update(dict1))

    def on_training_end(self) -> None:
        # unwrap cartpole environment from model
        env = self.model.env.envs[0].env
        while not env.state:
            env = env.env

        try:
            output_formats = self.logger.Logger.CURRENT.output_formats
            self.tb_formatter = next(
                formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))
            w = self.tb_formatter.writer
        except Exception:
            print(
                'Exception: No TensorBoardOutputFormat formatter found when trying to write hyperparameters in TBHparamCallback.on_training_end().')

        try:
            print('writing hparams to tb...')
            self.hparam_dict['hparam/avg_reward'] = env.accumulated_reward / env.num_timesteps
            self.hparam_dict['hparam/avg_abs_theta'] = env.accumulated_abs_theta / env.num_timesteps
            self.hparam_dict['hparam/avg_abs_x'] = env.accumulated_abs_x / env.num_timesteps
            w.add_hparams(hparam_dict=self.hparam_dict,
                          metric_dict={'hparam/avg_reward': np.array(env.accumulated_reward / env.num_timesteps),
                                       'hparam/avg_abs_theta': np.array(env.accumulated_abs_theta / env.num_timesteps),
                                       'hparam/avg_abs_x': np.array(env.accumulated_abs_x / env.num_timesteps)}
                          )
        except Exception as e:  # work on python 2.x
            print('Exception when writing hyperparameters + metrics' + str(e))

    def _on_step(self):
        pass


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0, every_steps=1000):
        super(TensorboardCallback, self).__init__(verbose)
        self.step = 0
        self.every_steps = every_steps

    def _on_step(self) -> bool:
        self.step += 1
        if self.step % self.every_steps == 0:
            self.model._dump_logs()
        return True


class CartPoleTensorboardCallback(BaseCallback):
    """
    Custom callback for plotting the cartpole state 4-tuple.
    """

    def __init__(self, log_dir, window_size=1000, verbose=0):
        super(CartPoleTensorboardCallback, self).__init__(verbose)

        self.window_size = window_size
        # window of x, x_dot, theta, theta_dot as empty lists
        self.x_wdw, self.x_dot_wdw, self.theta_p_is_south_wdw, self.theta_p_is_north_wdw, self.theta_dot_wdw = [], [], [], [], []
        # tb writer object
        self.writer = tf.summary.create_file_writer(log_dir)

    def _dump_logs(self):
        with self.writer.as_default():
            tf.summary.scalar("CartPole/x_std", np.std(self.x_wdw), step=self.num_timesteps)
            tf.summary.scalar("CartPole/x_dot_std", np.std(self.x_dot_wdw), step=self.num_timesteps)
            tf.summary.scalar("CartPole/theta_pi_is_south_std", np.std(self.theta_p_is_south_wdw),
                              step=self.num_timesteps)
            tf.summary.scalar("CartPole/theta_pi_is_north_std", np.std(self.theta_p_is_north_wdw),
                              step=self.num_timesteps)
            tf.summary.scalar("CartPole/theta_dot_std", np.std(self.theta_dot_wdw), step=self.num_timesteps)

            tf.summary.scalar("CartPole/x_mean", np.mean(self.x_wdw), step=self.num_timesteps)
            tf.summary.scalar("CartPole/x_dot_mean", np.mean(self.x_dot_wdw), step=self.num_timesteps)
            tf.summary.scalar("CartPole/theta_pi_is_south_mean", np.mean(self.theta_p_is_south_wdw),
                              step=self.num_timesteps)
            tf.summary.scalar("CartPole/theta_pi_is_north_mean", np.mean(self.theta_p_is_north_wdw),
                              step=self.num_timesteps)
            tf.summary.scalar("CartPole/theta_dot_mean", np.mean(self.theta_dot_wdw), step=self.num_timesteps)

            self.writer.flush()

    def _on_step(self) -> bool:
        # unwrap cartpole environment from model
        self.env = self.model.env.envs[0].env
        while not self.env.state:
            env = env.env

        x, x_dot, theta_pi_is_south, theta_dot = self.env.state

        theta_pi_is_north = ((theta_pi_is_south + math.pi) % (2 * math.pi))

        # add to windows
        self.x_wdw += [x]
        self.x_dot_wdw += [x_dot]
        self.theta_p_is_south_wdw += [theta_pi_is_south]
        self.theta_p_is_north_wdw += [theta_pi_is_north]
        self.theta_dot_wdw += [theta_dot]

        # cut windows to window size
        self.x_wdw = self.x_wdw[-self.window_size:]
        self.x_dot_wdw = self.x_dot_wdw[-self.window_size:]
        self.theta_p_is_south_wdw = self.theta_p_is_south_wdw[-self.window_size:]
        self.theta_p_is_north_wdw = self.theta_p_is_north_wdw[-self.window_size:]
        self.theta_dot_wdw = self.theta_dot_wdw[-self.window_size:]

        # every window_size steps dump to logger
        if self.num_timesteps % self.window_size == 0:
            self._dump_logs()

        return True
