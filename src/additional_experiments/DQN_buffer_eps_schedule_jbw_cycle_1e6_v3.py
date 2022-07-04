from __future__ import absolute_import, division, print_function
import sys, os

sys.path.append(os.path.join(sys.path[0], '../'))
from utility import util
import torch
import torch
import itertools as it
import custom_environments
from custom_wrappers import VisionWrapper, SoftHorizonWrapper, InfoWrapper
from custom_models import VisionCNN
from custom_callbacks import TBHparamCallback

sys.modules['jbw.environments'] = custom_environments
import gym
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from custom_dqn_exploration_schedule import DQN

seeds = [1, 2, 3, 4, 5]  # multiple runs, one for each of these seeds
hp_settings = {
    'soft_horizon': [1e4],  # logical episodes every soft_horizon steps without resetting env
    'wrapper_scheduler_modulo': [5e5],  # toggle wrappers at this interval (in steps)
    'policy_batch_size': [32],
    'policy_features_dim': [128],
    'buffer_size': [int(f) for f in [1e6, 1e5, 1e4, 1e3, 5e3]]
}

env_name = 'JBW-cycle-1e6-v3'  # the gym envrionment name
iterations = 2e6  # steps per run
save_freq = 2.5e7  # steps between saves
experiment_name = 'DQN_buffer_eps_schedule_' + env_name  # used for creating log and save paths
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


def exploration_schedule(step):
    exp_initial_steps = 1.0
    exp_fraction = 0.1
    exp_final_eps = 0.05

    s = step % 1e6
    return max(exp_initial_steps - ((s / 1e6) / exp_fraction), exp_final_eps)


keys = sorted(hp_settings)
hp_combinations = it.product(*(hp_settings[k] for k in keys))

for seed in seeds:
    for hp_values in hp_combinations:
        hp_dict = dict(zip(keys, hp_values))
        print(hp_dict)

        torch.cuda.empty_cache()

        # create string that holds what is unique to this combination
        unique_params = util.get_unique_params(hp_settings, hp_dict)
        # append it as a subfolder to experiment
        log_dir = '../tensorboard_logs/' + experiment_name + '/' + unique_params + '/seed_' + str(seed)
        save_dir = '../saves/' + experiment_name + '/' + unique_params + '/seed_' + str(seed)

        # skip existing hyperparameter experiments
        if os.path.exists(log_dir) and os.path.exists(save_dir):
            continue

        # store hp_dict to some json/yml file
        util.yaml_save(config=hp_dict, path=save_dir + '/hp_config.yaml')

        print('unique_params:', unique_params)


        print('new training run for seed', seed)
        print('log_dir', log_dir)
        print('save_dir', save_dir)

        # Create training environment and train
        print('creating training env...')
        env = gym.make(env_name)
        env = VisionWrapper(env)

        env = InfoWrapper(env)
        env = SoftHorizonWrapper(env, soft_horizon=hp_dict['soft_horizon'], max_resets=1)

        print('creating model...')
        policy_kwargs = dict(
            features_extractor_class=VisionCNN,
            features_extractor_kwargs=dict(features_dim=hp_dict['policy_features_dim']),
        )

        model = DQN('CnnPolicy', env, policy_kwargs=policy_kwargs, verbose=1,
                    tensorboard_log=log_dir, seed=seed, buffer_size=hp_dict['buffer_size'],
                    batch_size=hp_dict['policy_batch_size'], exploration_schedule=exploration_schedule)

        checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=save_dir,
                                                 name_prefix='', verbose=1)
        callback_list = CallbackList([checkpoint_callback, TBHparamCallback(hparam_dict=hp_dict)])

        print('training...')
        model.learn(total_timesteps=int(iterations), callback=callback_list, log_interval=1)
