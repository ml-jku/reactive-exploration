import os
import sys
sys.path.insert(1, 'src/')
import wandb
import hydra
import omegaconf
import stable_baselines3 as sb3
from pathlib import Path
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from src import custom_environments, cartpole
sys.modules['jbw.environments'] = custom_environments
sys.modules['cartpole'] = cartpole
import gym
from src.curiosity_ppo import PPO
from src.dqn_per import PrioritizedExperienceReplayDQN
from src.curiosity_dqn import DCQN
from src.soft_q_learning import SQL
from src.curiosity_sql import CSQL
from src.rnd import RNDFactory, RNDModelFactory, NoveldDFactory
from src.ride import RIDEFactory
from src.icmppo.reporters import TensorBoardReporter
from src.icmppo.curiosity.icm import ICM, CNNICMModel, MlpICMModel
from src.rewardprediction import RewardPredictor, RewardModel, MlpRewardModel
from src.custom_wrappers import VisionWrapper, SoftHorizonWrapper, InfoWrapper, CyclicScheduler, \
    SwapperGivenPosition, ObsRotator, EnvListScheduler, StoreTrajectory, ColourSwapper
from src.custom_models import VisionCNN


def make_colours():
    w = [0, 0, 0]
    black = [1, 1, 1]
    r = [1, 0, 0]
    g = [0, 1, 0]
    blue = [0, 0, 1]
    m = [1, 0, 1]
    c = [0, 1, 1]
    y = [1, 1, 0]
    old_colours = [[w, g, r], [w, g, r], [w, g, r], [w, g, r]]
    new_colours = [[g, r, w], [r, black, g], [c, y, m], [black, r, c]]
    return old_colours, new_colours


def make_env(env_name="test", step_modulo=5e5,
             use_vision_wrapper=True, use_info_wrapper=True,
             use_colour_swapper=False, use_rotator=False, use_green_white_swapper=False,
             soft_horizon=None, cartpole_dicts=None, recovery_start=None):
    print('Making training env...')
    if "cartpole" in env_name.lower():
        cartpole_dicts = omegaconf.OmegaConf.to_container(cartpole_dicts, resolve=True)
        env = gym.make(env_name, param_dict=cartpole_dicts, change_interval=step_modulo)
    else:
        env = gym.make(env_name)
    if use_vision_wrapper:
        env = VisionWrapper(env)

    if use_colour_swapper:
        old_colours, new_colours = make_colours()
        handle = SwapperGivenPosition(env, old_colours=old_colours, new_colours=new_colours)
        env = CyclicScheduler(env, [env, handle], step_modulo=float(step_modulo))
        if recovery_start:
            env = EnvListScheduler(env, handle_dict={0: env, recovery_start: handle})
    if use_green_white_swapper:
        handle = ColourSwapper(env, [[0, 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 0, 0]])
        env = CyclicScheduler(env, [env, handle], step_modulo=float(step_modulo))
    if use_rotator:
        handle = ObsRotator(env, rotate_by=2)
        env = CyclicScheduler(env, [env, handle], step_modulo=float(step_modulo))
    if use_info_wrapper:
        env = InfoWrapper(env)
    if soft_horizon:
        env = SoftHorizonWrapper(env, soft_horizon=float(soft_horizon), max_resets=1)
    return env


def make_agent(env, agent_params, log_dir=None, seed=None):
    print('Making Agent...')
    is_cnn_policy = agent_params.policy in ["CnnPolicy", "SQLCnnPolicy", "DCQNCnnPolicy", "CSQLCnnPolicy"]
    if is_cnn_policy:
        policy_kwargs = dict(
            features_extractor_class=VisionCNN,
            features_extractor_kwargs=dict(features_dim=128),
        )
    else:
        policy_kwargs = None

    exploration_factory = None
    reward_predictor_factory = None
    if "exploration_params" in agent_params:
        exploration_params = omegaconf.OmegaConf.to_container(agent_params.exploration_params, resolve=True)
        kind = exploration_params.pop("kind")
        model_params = exploration_params.pop("model_params") if "model_params" in exploration_params else {}
        exploration_reporter = TensorBoardReporter(logdir=log_dir + '/' + kind) if log_dir else None

        if kind == "RND":
            exp_model_factory_kind = "cnn" if is_cnn_policy else "mlp"
            exploration_factory = RNDFactory(RNDModelFactory(kind=exp_model_factory_kind, **model_params),
                                             reporter=exploration_reporter,
                                             **exploration_params)
        elif kind == "NovelD":
            exp_model_factory_kind = "cnn" if is_cnn_policy else "mlp"
            exploration_factory = NoveldDFactory(RNDModelFactory(kind=exp_model_factory_kind, **model_params),
                                                 reporter=exploration_reporter,
                                                 **exploration_params)
        elif kind == "ICM":
            exp_model = CNNICMModel if is_cnn_policy else MlpICMModel
            exploration_factory = ICM.factory(exp_model.factory(), reporter=exploration_reporter,
                                              **exploration_params)
        elif kind == "RIDE":
            exp_model = CNNICMModel if is_cnn_policy else MlpICMModel
            exploration_factory = RIDEFactory(exp_model.factory(), reporter=exploration_reporter,
                                              **exploration_params)
        else:
            raise NotImplementedError()

    if "use_reward_predictor" in agent_params and agent_params.use_reward_predictor:
        reward_model_params = omegaconf.OmegaConf.to_container(agent_params.reward_model_params, resolve=True) \
            if "reward_model_params" in agent_params else {}
        reward_model_reporter = TensorBoardReporter(logdir=log_dir + '/RewardModel') if log_dir else None
        reward_model = RewardModel if is_cnn_policy else MlpRewardModel
        reward_predictor_factory = RewardPredictor.factory(reward_model.factory(), reporter=reward_model_reporter,
                                                           **reward_model_params)

    if "kind" in agent_params and agent_params.kind == "DQN":
        agent = sb3.DQN(policy=agent_params["policy"], env=env, policy_kwargs=policy_kwargs, verbose=1,
                        tensorboard_log=log_dir, seed=seed,
                        buffer_size=int(agent_params.buffer_size) if "buffer_size" in agent_params else None)
    elif "kind" in agent_params and agent_params.kind == "DQN_PER":
        agent = PrioritizedExperienceReplayDQN(policy=agent_params["policy"], env=env, policy_kwargs=policy_kwargs,
                                               verbose=1, tensorboard_log=log_dir, seed=seed,
                                               buffer_size=int(agent_params.buffer_size) if "buffer_size" in agent_params else None,
                                               priority_replay_alpha=float(agent_params.alpha) if "alpha" in agent_params else 0.6,
                                               priority_replay_beta=float(agent_params.beta) if "beta" in agent_params else 0.4,
                                               priority_replay_beta_increment_per_sampling=float(agent_params.beta_increment) if "beta_increment" in agent_params else 0.001)
    elif "kind" in agent_params and agent_params.kind == "SQL":
        agent = SQL(policy=agent_params["policy"], env=env, policy_kwargs=policy_kwargs, verbose=1,
                    tensorboard_log=log_dir, seed=seed,
                    buffer_size=int(agent_params.buffer_size) if "buffer_size" in agent_params else None)
    elif "exploration_params" in agent_params:
        if "kind" in agent_params and agent_params.kind == "DCQN":
            agent = DCQN(policy=agent_params["policy"], env=env, policy_kwargs=policy_kwargs, verbose=1,
                         tensorboard_log=log_dir, seed=seed,
                         buffer_size=int(agent_params.buffer_size) if "buffer_size" in agent_params else None,
                         curiosity_factory=exploration_factory,
                         reward_predictor_factory=reward_predictor_factory)
        elif "kind" in agent_params and agent_params.kind == "CSQL":
            agent = CSQL(policy=agent_params["policy"], env=env, policy_kwargs=policy_kwargs, verbose=1,
                         tensorboard_log=log_dir, seed=seed,
                         buffer_size=int(agent_params.buffer_size) if "buffer_size" in agent_params else None,
                         curiosity_factory=exploration_factory,
                         reward_predictor_factory=reward_predictor_factory)
        else:
            ent_coef = agent_params.ent_coef if "ent_coef" in agent_params else 0.0
            agent = PPO(policy=agent_params["policy"], env=env, policy_kwargs=policy_kwargs, verbose=1,
                        tensorboard_log=log_dir, seed=seed, curiosity_factory=exploration_factory,
                        reward_predictor_factory=reward_predictor_factory, ent_coef=ent_coef)
    else:
        ent_coef = agent_params.ent_coef if "ent_coef" in agent_params else 0.0
        agent = sb3.PPO(policy=agent_params["policy"], env=env, policy_kwargs=policy_kwargs, verbose=1,
                        tensorboard_log=log_dir, seed=seed, ent_coef=ent_coef)
    return agent


def make_callbacks(save_dir=None, save_freq=None):
    print('Making callbacks...')
    callbacks = []
    if save_dir and save_freq:
        callbacks.append(CheckpointCallback(save_freq=save_freq, save_path=save_dir, name_prefix='', verbose=1))
    return CallbackList(callbacks)


def setup_wandb(config):
    print("Setting up logging to Weights & Biases.")
    config_dict = omegaconf.OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    # hydra changes working directories
    log_dir = str(Path.joinpath(Path(os.getcwd()), config.run_params.log_dir))
    # make "wandb" path, otherwise WSL might block writing to dir
    wandb_path = Path.joinpath(Path(log_dir), "wandb")
    wandb_path.mkdir(exist_ok=True, parents=True)
    wandb.login()
    # tracks everything that TensorBoard tracks
    wandb.tensorboard.patch(root_logdir=log_dir)
    wandb_run = wandb.init(project="lifelong-rl", name=log_dir,
                           dir=log_dir, save_code=False, config=config_dict)
    wandb_run.tags = wandb_run.tags + tuple(make_wandb_tags(config_dict))
    print(f"Writing Weights & Biases logs to: {str(wandb_path)}")
    return wandb_run


def make_wandb_tags(config):
    tags = []
    for key in ["experiment_name", "seed"]:
        if key in config:
            val = config[key]
            if key == "seed":
                val = "seed_" + str(val)
            tags.append(val)
    return tags


@hydra.main(config_path="configs", config_name="config")
def main(config):
    print("Config: ", config)
    if config.use_wandb:
        setup_wandb(config)

    env = make_env(**config.env_params)
    if config.store_trajectory:
        env = StoreTrajectory(env, config.steps, start_step=config.start_recording, end_step=config.stop_recording)
    agent = make_agent(env=env, agent_params=config.agent_params, log_dir=config.run_params.log_dir, seed=config.seed)
    callbacks = make_callbacks(config.run_params.save_dir, **config.callback_params)
    print('Starting training...')
    agent.learn(total_timesteps=int(config.steps), callback=callbacks, log_interval=1)
    if config.use_wandb:
        # necessary for Hydra multiruns
        wandb.finish()
        wandb.tensorboard.unpatch()


if __name__ == "__main__":
    main()
