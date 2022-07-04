from __future__ import absolute_import, division, print_function
import torch
import sys, os

sys.path.append(os.path.join(sys.path[0], '../'))
import itertools as it
from utility import util
import torch
from custom_wrappers import InfoWrapper, SoftHorizonWrapper
from custom_callbacks import CartPoleTensorboardCallback
from custom_callbacks import CartPoleTBHparamCallback

import gym
from cartpole import CartPoleEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList

seeds = [1, 2, 3, 4, 5]  # multiple runs, one for each of these seeds
hp_settings = {
    'soft_horizon': [1e4],  # logical episodes every soft_horizon steps without resetting env
    'ppo_batch_size': [64],
}
env_configs = {
    'stop_at_x_threshold': [True],
    'modulo_theta': [True],
    'reward_function': ['variable'],
    'rew_xc': [1 / 11.52],
    'rew_theta_c': [1 / 288],
    'reset_pole_on_drop': [True],
    'length': [1],
    'masscart': [2],
    'force_mag': [-10]
}

env_name = 'InfHCartPole-v1'  # the gym envrionment name
iterations = 1e6  # steps per run
save_freq = 2.5e7  # steps between saves
experiment_name = 'PPO_stationary_' + env_name + '/len1massc2forcemag-10'  # used for creating log and save paths
device_str = 'cuda:0'  # e.g. cuda:0

if torch.cuda.is_available():
    torch.cuda.set_device(device_str)
    print('cuda current device set to:', torch.cuda.current_device())
else:
    print('gpu/cuda not available')

print('seeds are', seeds)
print('num steps set to', iterations)
print('using environment', env_name)
print('experiment name:', experiment_name)
# storing hp settings with all array entries
util.yaml_save(config=hp_settings, path='../saves/' + experiment_name + '/hp_configs.yaml')
util.yaml_save(config=env_configs, path='../saves/' + experiment_name + '/env_configs.yaml')

# all combinations of hparams
hp_keys = sorted(hp_settings)
hp_combinations = it.product(*(hp_settings[k] for k in hp_keys))
# all combiantions of env configurations
env_keys = sorted(env_configs)
env_hp_combinations = it.product(*(env_configs[k] for k in env_keys))

env_hp_combinations, env_hp_combinations_backup = it.tee(env_hp_combinations)
hp_combinations, hp_combinations_backup = it.tee(hp_combinations)

combinations = []

for seed in seeds:
    env_hp_combinations, env_hp_combinations_backup = it.tee(env_hp_combinations_backup)
    for env_cfg in env_hp_combinations:
        hp_combinations, hp_combinations_backup = it.tee(hp_combinations_backup)
        for hp_dict in hp_combinations:
            hp_dict = dict(zip(hp_keys, hp_dict))
            env_dict = dict(zip(env_keys, env_cfg))
            combinations += [{'seed': seed, 'env_cfg': env_dict, 'hp_dict': hp_dict, 'hp_settings': hp_settings,
                              'env_configs': env_configs}]


def run_training_instance(args):
    print(args)
    seed, env_cfg, hp_dict, hp_settings, env_configs = args['seed'], args['env_cfg'], args['hp_dict'], args[
        'hp_settings'], args['env_configs']
    print(hp_dict)
    print(env_cfg)
    print(seed)

    torch.cuda.empty_cache()

    # create string that holds what is unique to this combination
    hp_unique_params = util.get_unique_params(hp_settings, hp_dict)
    env_unique_params = util.get_unique_params(env_configs, env_cfg)
    # append it as a subfolder to experiment
    log_dir = '../tensorboard_logs/' + experiment_name + '/' + env_unique_params + '/' + hp_unique_params + '/seed_' + str(
        seed)
    save_dir = '../saves/' + experiment_name + '/' + env_unique_params + '/' + hp_unique_params + '/seed_' + str(seed)

    # skip existing hyperparameter experiments
    if os.path.exists(log_dir) and os.path.exists(save_dir):
        print('skipping experiment - found existing files for', save_dir)
        return

    # store hp_dict to some json/yml file
    util.yaml_save(config=hp_dict, path=save_dir + '/hp_config.yaml')
    # store env_cfg to some json/yml file
    util.yaml_save(config=env_cfg, path=save_dir + '/env_config.yaml')

    print('unique hyperparams:', hp_unique_params)
    print('unique env config parameters:', env_unique_params)
    print('new training run for seed', seed)
    print('log_dir', log_dir)
    print('save_dir', save_dir)

    # Create training environment and train
    print('creating training env...')
    env = gym.make(env_name, param_dict=env_cfg)
    env = InfoWrapper(env)
    env = SoftHorizonWrapper(env, soft_horizon=hp_dict['soft_horizon'], max_resets=1)

    print('creating model...')
    model = PPO('MlpPolicy', env, verbose=1, batch_size=hp_dict['ppo_batch_size'],
                tensorboard_log=log_dir, seed=seed)
    callback_list = CallbackList([CartPoleTensorboardCallback(log_dir=log_dir + "/CartPole_" + str(seed)),
                                  CartPoleTBHparamCallback(hparam_dict=hp_dict, env_config=env_cfg)])

    print('training...')
    model.learn(total_timesteps=int(iterations), callback=callback_list)


if __name__ == '__main__':

    for args in combinations:
        run_training_instance(args)
