import gym
import numpy as np
import pickle
import os


class StoreTrajectory(gym.Wrapper):
    def __init__(self, env, n_steps, trajectory_log_dir='trajectory', start_step=0, end_step=-1):
        super(StoreTrajectory, self).__init__(env)
        # in case only a window between two step numbers is logged
        self.start_step = int(start_step)
        self.end_step = int(end_step) if int(end_step) != -1 else int(n_steps)
        trajectory_log_dir = trajectory_log_dir + '_' + str(self.start_step) + '-' + str(self.end_step)
        # prepare empty numpy arrays
        self.observations = np.empty(shape=(int(self.end_step - self.start_step),) + env.observation_space.shape)
        self.actions = np.empty(shape=(int(self.end_step - self.start_step),) + env.action_space.shape)
        self.rewards = np.empty(shape=(int(self.end_step - self.start_step),))
        # anything that is not logged should be nan
        self.observations[:] = np.nan
        self.actions[:] = np.nan
        self.rewards[:] = np.nan

        self.steps = 0
        self.trajectory_log_dir = trajectory_log_dir
        self.n_steps = n_steps
        # idx=0 is treated as the reset where only an observation and action are present but no reward
        # idx=-1 is treated as the last step, where an observation and reward are returned but noaction is chosen
        self.idx = 1
        self.last_obs = None

    def reset(self):
        obs = self.env.reset()
        if self.start_step <= self.steps <= self.end_step:
            self.last_obs = obs
        self.steps += 1
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.steps == int(3e4):
            print()
        if self.idx < len(self.observations) and self.start_step <= self.steps < self.end_step:
            self.actions[self.idx - 1] = action
            self.observations[self.idx - 1] = self.last_obs
            self.observations[self.idx] = obs
            self.rewards[self.idx] = reward
            self.idx += 1
        if self.steps == self.end_step:
            if not os.path.exists(self.trajectory_log_dir):
                os.makedirs(self.trajectory_log_dir)
            with open(os.path.join(self.trajectory_log_dir, 'observations.pickle'), 'wb') as handle:
                pickle.dump(self.observations, handle)
            with open(os.path.join(self.trajectory_log_dir, 'actions.pickle'), 'wb') as handle:
                pickle.dump(self.actions, handle)
            with open(os.path.join(self.trajectory_log_dir, 'rewards.pickle'), 'wb') as handle:
                pickle.dump(self.rewards, handle)
            self.idx += 1

        self.steps += 1
        self.last_obs = obs
        return obs, reward, done, info


class SwapperGivenPosition(gym.Wrapper):
    def __init__(self, env, old_colours, new_colours, interpolate=False, cycle_steps=0):
        super(SwapperGivenPosition, self).__init__(env)
        assert (len(old_colours) == 4)
        assert (len(new_colours) == 4)
        self.old_colours = old_colours
        self.new_colours = new_colours
        self.interpolate = interpolate
        self.cycle_steps = cycle_steps
        self.steps = 0

    def _mix_obs(self, obs0, obs1):
        # if pre half cycle
        if self.steps % self.cycle_steps < (self.cycle_steps // 2):
            # move obs0_ratio down from 1 and obs1_ratio up from 0
            obs1_ratio = (self.steps % (self.cycle_steps // 2)) / (self.cycle_steps // 2)
            obs0_ratio = 1.0 - obs1_ratio
        # if post half cycle
        else:
            obs0_ratio = (self.steps % (self.cycle_steps // 2)) / (self.cycle_steps // 2)
            obs1_ratio = 1.0 - obs0_ratio
        # move obs0_ratio up from 0 and obs1_ratio down from 1

        return obs0 * obs0_ratio + obs1 * obs1_ratio

    def swap(self, obs):
        w = obs.shape[0]
        h = obs.shape[1]
        dir_masks = [[np.all(obs == old, axis=2) for old in c] for c in self.old_colours]
        dir_masks = np.array(dir_masks)

        m1 = np.zeros((w, h))
        m1[:w // 2, :h // 2] = 1
        dir_masks[0] = np.logical_and(dir_masks[0], m1)

        m2 = np.zeros((w, h))
        m2[w // 2:, :h // 2] = 1
        dir_masks[1] = np.logical_and(dir_masks[1], m2)

        m3 = np.zeros((w, h))
        m3[:w // 2, h // 2:] = 1
        dir_masks[2] = np.logical_and(dir_masks[2], m3)

        m4 = np.zeros((w, h))
        m4[w // 2:, h // 2:] = 1
        dir_masks[3] = np.logical_and(dir_masks[3], m4)

        for masks, new_dir in zip(dir_masks, self.new_colours):
            for mask, new in zip(masks, new_dir):
                obs[mask] = new

        return obs

    def reset(self):
        obs0 = self.env.reset()
        obs1 = self.swap(obs0.copy())
        if self.interpolate:
            return self._mix_obs(obs0, obs1)
        else:
            return obs1

    def step(self, action):
        self.steps += 1
        obs0, reward, done, info = self.env.step(action)
        obs1 = self.swap(obs0.copy())
        if self.interpolate:
            return self._mix_obs(obs0, obs1), reward, done, info
        else:
            return obs1, reward, done, info


class ColourSwapper(gym.Wrapper):
    """
    Swaps pixel colours in observations
    old_colours: the original pixel colours that should be swapped. old_colours[i] will be swapped for new_colours[i]
    new_colours: the new pixel colours that should replace old ones. old_colours[i] will be swapped for new_colours[i]
    """

    def __init__(self, env, old_colours, new_colours):
        super(ColourSwapper, self).__init__(env)
        self.old_colours = old_colours
        self.new_colours = new_colours

    def swap(self, obs):
        masks = []
        for old in self.old_colours:
            masks.append(np.all(obs == old, axis=2))

        for mask, new in zip(masks, self.new_colours):
            obs[mask] = new

        return obs

    def reset(self):
        obs = self.env.reset()
        obs = self.swap(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.swap(obs)
        return obs, reward, done, info


class EnvListScheduler(gym.Wrapper):
    """
    Switches through a list of wrappers / environment handles. Given a dictionary of steps and environments
    the scheduler swaps between the environments given the current step.

    handle_dict: {step: environment, ...} A dict with keys=steps and values=environment handles. At
    each step the dict is checked for an entry with key step and the scheduler changes to the corresponding
    environment handle value.
    """

    def __init__(self, env, handle_dict):
        super(EnvListScheduler, self).__init__(env)
        self.steps = 0
        self.handle_dict = handle_dict

    def swap_handles(self):
        self.env_handle = self.handle_dict[self.steps]

    def reset(self):
        if self.steps in self.handle_dict:
            self.swap_handles()

        obs = self.env_handle.reset()
        return obs

    def step(self, action):
        if self.steps in self.handle_dict:
            self.swap_handles()

        obs, reward, done, info = self.env_handle.step(action)

        self.steps += 1
        return obs, reward, done, info


class CyclicScheduler(gym.Wrapper):
    """
    Cycles a list of wrappers / environment handles. Uses handle_list[0] for step and reset function until step_modulo
    steps, then uses handle_list[1] etc and cycles back to handle_list[0] once the last has been cycled through.

    handle_list: A list of handles that is going to be used to call step and reset methods.
    step_modulo: Amount of steps until the next handle in the list is switched to.
    """

    def __init__(self, env, handle_list, step_modulo):
        super(CyclicScheduler, self).__init__(env)
        self.steps = 0
        self.step_modulo = step_modulo
        self.handle_list = handle_list
        self.env_handle = self.handle_list[0]
        self.active_handle_num = 0

    def swap_handles(self):
        self.active_handle_num = (self.active_handle_num + 1) % len(self.handle_list)
        self.env_handle = self.handle_list[self.active_handle_num]

    def reset(self):
        if self.steps % self.step_modulo == 0:
            self.swap_handles()

        obs = self.env_handle.reset()
        return obs

    def step(self, action):
        if self.steps % self.step_modulo == 0:
            self.swap_handles()

        obs, reward, done, info = self.env_handle.step(action)

        self.steps += 1
        return obs, reward, done, info


class ObsRotator(gym.Wrapper):
    """
    Rotates entire observation space k times 90 degrees. If rotate_by is None, k is increased/decreased whenever the agent takes its rotate actions. Otherwise it is statically rotated by k.
    Intended to be used to modify transition probabilities SxAxS by enabling/disabling this rotator after some steps

    rotate_by: If this is none, the amount by which the observations are rotated is altered everytime the agent turns left or right. This happens in such a way that the observation space never actually rotates and stays still, while the default for JBW is otherwise that the rotation space rotates with the agent.
    """

    def __init__(self, env, rotate_by=None):
        super(ObsRotator, self).__init__(env)
        self.rotate_by = rotate_by
        if rotate_by:
            self.n_rot90 = rotate_by
        else:
            self.n_rot90 = 0

    def reset(self):
        obs = self.env.reset()
        obs = np.rot90(obs, k=self.n_rot90)
        return obs

    def step(self, action):
        if not self.rotate_by:
            if action == [1]:
                self.n_rot90 = (self.n_rot90 + 1) % 4
            elif action == [2]:
                self.n_rot90 = (self.n_rot90 - 1) % 4

        obs, reward, done, info = self.env.step(action)
        obs = np.rot90(obs, k=self.n_rot90)

        return obs, reward, done, info


class InfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super(InfoWrapper, self).__init__(env)

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["episode"] = {"r": reward, "l": 1}
        return obs, reward, done, info


class SoftHorizonWrapper(gym.Wrapper):
    """
    Acts as if the environment is episodic while never resetting it apart from the very first reset for initialization.
    This way the aspects of the API which are built around episodic RL still work as if the env was episodic, while the
    env is actually used in a non-episodic way, without any resets.

    :param env: (gym.Env) Gym environment that 	will be wrapped
    """

    def __init__(self, env, soft_horizon=10000, max_resets=1):
        super(SoftHorizonWrapper, self).__init__(env)
        self.steps = 0
        self.n_resets = 0
        self.max_resets = max_resets
        self.soft_horizon = soft_horizon
        self.last_obs = None
        self.r = 0

    def _really_reset(self):
        obs = self.env.reset()
        self.last_obs = obs
        return obs

    def reset(self):
        """
        Only calls the env reset until a reset has been called max_reset times. After that only returns the last observation
        """
        self.n_resets += 1
        if self.n_resets <= self.max_resets:
            return self._really_reset()
        return self.last_obs

    def step(self, action):
        self.steps += 1
        obs, reward, done, info = self.env.step(action)
        self.r += reward
        self.last_obs = obs
        if self.steps % self.soft_horizon == 0:
            done = True
            # info["episode"] = {"r": self.r, "l": self.soft_horizon}
            self.r = 0
            self.l = 0
        return obs, reward, done, info


class VisionWrapper(gym.Wrapper):
    """
    Takes the JBW environments' obs and uses only the 'vision' aspect, to make it work with default models.
    :param env: (gym.Env) Gym environment that 	will be wrapped
    """

    def __init__(self, env, channels_first=False):
        # Call the parent constructor, so we can access self.env later
        vision_range = env.sim_config.vision_range
        vision_dim = len(env.sim_config.items[0].color)
        self.channels_first = channels_first
        if channels_first:
            vision_shape = [
                vision_dim,
                2 * vision_range + 1,
                2 * vision_range + 1]
        else:
            vision_shape = [
                2 * vision_range + 1,
                2 * vision_range + 1,
                vision_dim]
        min_vision = 0 * np.ones(vision_shape)
        max_vision = 255 * np.ones(vision_shape)
        env.observation_space = gym.spaces.Box(low=min_vision, high=max_vision, dtype=np.uint8)
        super(VisionWrapper, self).__init__(env)

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()
        obs = obs['vision']
        if self.channels_first:
            obs = np.moveaxis(obs, 2, 0)
        # add [n_envs, ] to shape
        # np.expand_dims(obs, 0)
        return obs

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        obs, reward, done, info = self.env.step(action)
        obs = obs['vision']
        if self.channels_first:
            obs = np.moveaxis(obs, 2, 0)
        # new_obs = [] + obs['vision'] + obs['scent'] + obs['moved']
        # print(type(obs['vision']), obs['vision'].shape)
        # print(type(obs['scent']), obs['scent'].shape)
        # print(type(obs['moved']))

        # add [n_envs, ] to shape
        # np.expand_dims(obs, 0)
        return obs, reward, done, info
