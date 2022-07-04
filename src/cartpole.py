"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class CartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, param_dict, change_interval=None, interpolate=False):

        assert (isinstance(param_dict, list) and change_interval) or (
                    not isinstance(param_dict, list) and not change_interval)
        self.change_interval = change_interval
        self.interpolate = interpolate

        if isinstance(param_dict, list):
            self.param_dicts = param_dict
            self.param_dict_i = 0
            self.params = self.param_dicts[self.param_dict_i]
        else:
            self.params = param_dict

        self.gravity = self.params['gravity'] if 'gravity' in self.params else 9.8
        self.masscart = self.params['masscart'] if 'masscart' in self.params else 1.0
        self.masspole = self.params['masspole'] if 'masspole' in self.params else 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = self.params['length'] if 'length' in self.params else 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = self.params['force_mag'] if 'force_mag' in self.params else 10.0
        self.tau = self.params['tau'] if 'tau' in self.params else 0.02  # seconds between state updates
        self.kinematics_integrator = self.params[
            'kinematics_integrator'] if 'kinematics_integrator' in self.params else 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = self.params[
            'theta_threshold_radians'] if 'theta_threshold_radians' in self.params else 12 * 2 * math.pi / 360
        self.x_threshold = self.params['x_threshold'] if 'x_threshold' in self.params else 2.4
        # self.temp_threshold = 3
        # self.xacc_threshold = 1

        self.stop_at_x_threshold = self.params['stop_at_x_threshold'] if 'stop_at_x_threshold' in self.params else True
        self.modulo_theta = self.params['modulo_theta'] if 'modulo_theta' in self.params else True
        self.reward_function = self.params['reward_function'] if 'reward_function' in self.params else 'variable'
        # # https://towardsdatascience.com/infinite-steps-cartpole-problem-with-variable-reward-7ad9a0dcf6d0
        # original parameters: reward = (1 - (x**2) / 11.52 + (theta ** 2) / 288)
        self.rew_xc = self.params['rew_xc'] if 'rew_xc' in self.params else 1 / 11.52
        self.rew_theta_c = self.params['rew_theta_c'] if 'rew_theta_c' in self.params else 1 / 288

        self.reset_pole_on_drop = self.params['reset_pole_on_drop'] if 'reset_pole_on_drop' in self.params else False

        self.accumulated_reward = 0
        self.accumulated_abs_theta = 0
        self.accumulated_abs_x = 0
        self.num_timesteps = 0

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        if self.stop_at_x_threshold:
            x_high = self.x_threshold * 2
        else:
            x_high = np.finfo(np.float32).max,
        high = np.array([x_high,
                         np.finfo(np.float32).max,
                         #  self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def update_environment_parameters(self, params):
        self.gravity = params['gravity'] if 'gravity' in params else 9.8
        self.masscart = params['masscart'] if 'masscart' in params else 1.0
        self.masspole = params['masspole'] if 'masspole' in params else 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = params['length'] if 'length' in params else 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = params['force_mag'] if 'force_mag' in params else 10.0
        self.tau = params['tau'] if 'tau' in params else 0.02  # seconds between state updates
        self.kinematics_integrator = params['kinematics_integrator'] if 'kinematics_integrator' in params else 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = params[
            'theta_threshold_radians'] if 'theta_threshold_radians' in params else 12 * 2 * math.pi / 360
        self.x_threshold = params['x_threshold'] if 'x_threshold' in params else 2.4
        # self.temp_threshold = 3
        # self.xacc_threshold = 1

        self.stop_at_x_threshold = params['stop_at_x_threshold'] if 'stop_at_x_threshold' in params else True
        self.modulo_theta = params['modulo_theta'] if 'modulo_theta' in params else True
        self.reward_function = params['reward_function'] if 'reward_function' in params else 'variable'
        # # https://towardsdatascience.com/infinite-steps-cartpole-problem-with-variable-reward-7ad9a0dcf6d0
        # original parameters: reward = (1 - (x**2) / 11.52 + (theta ** 2) / 288)
        self.rew_xc = params['rew_xc'] if 'rew_xc' in params else 1 / 11.52
        self.rew_theta_c = params['rew_theta_c'] if 'rew_theta_c' in params else 1 / 288

        self.reset_pole_on_drop = params['reset_pole_on_drop'] if 'reset_pole_on_drop' in params else False

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        if self.stop_at_x_threshold:
            x_high = self.x_threshold * 2
        else:
            x_high = np.finfo(np.float32).max,
        high = np.array([x_high,
                         np.finfo(np.float32).max,
                         #  self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        ##############
        # kinematics #
        ##############

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # xacc = min(xacc, self.xacc_threshold)
        # xacc = max(xacc, -self.xacc_threshold)
        if self.kinematics_integrator == 'euler':

            x = x + self.tau * x_dot

            if self.stop_at_x_threshold and abs(x) > self.x_threshold:
                x = min(x, self.x_threshold)
                x = max(x, -self.x_threshold)
                x_dot = 0
                xacc = 0
                temp = 0

            x_dot = x_dot + self.tau * xacc

            previous_theta = theta
            theta_before_modulo = (theta + self.tau * theta_dot)
            if self.modulo_theta:
                theta = (theta + self.tau * theta_dot) % (2 * math.pi)

            else:
                theta = (theta + self.tau * theta_dot)

            if self.reset_pole_on_drop:
                if (theta_dot > 0 and previous_theta < math.pi and theta_before_modulo >= math.pi) or (
                        theta_dot < 0 and previous_theta > math.pi and theta_before_modulo <= math.pi):
                    theta, theta_dot = self.np_random.uniform(low=-0.05, high=0.05, size=(2,))

            theta_dot = theta_dot + self.tau * thetaacc

        else:  # semi-implicit euler

            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot

            if self.stop_at_x_threshold and abs(x) > self.x_threshold:
                x = min(x, self.x_threshold)
                x = max(x, -self.x_threshold)
                x_dot = 0
                xacc = 0
                temp = 0

            theta_dot = theta_dot + self.tau * thetaacc
            previous_theta = theta
            theta_before_modulo = (theta + self.tau * theta_dot)
            if self.modulo_theta:
                theta = (theta + self.tau * theta_dot) % (2 * math.pi)

            else:
                theta = (theta + self.tau * theta_dot)

            if self.reset_pole_on_drop:
                if (theta_dot > 0 and previous_theta < math.pi and theta_before_modulo >= math.pi) or (
                        theta_dot < 0 and previous_theta > math.pi and theta_before_modulo <= math.pi):
                    theta, theta_dot = self.np_random.uniform(low=-0.05, high=0.05, size=(2,))

        self.state = (x, x_dot, theta, theta_dot)

        ###################
        # reward function #
        ###################

        if self.reward_function == 'original':
            reward = 1.0
            if abs(theta) > self.theta_threshold_radians:
                reward = 0.0
        elif self.reward_function == 'variable':
            # variable reward
            # https://towardsdatascience.com/infinite-steps-cartpole-problem-with-variable-reward-7ad9a0dcf6d0
            # https://github.com/tensorflow/agents/blob/master/tf_agents/environments/suite_gym.py
            # original parameters: reward = (1 - (x**2) / 11.52 + (theta ** 2) / 288)
            # ( (theta + math.pi) % (2 * math.pi) ) is adjusted theta such that balanced pole facing north = 1*pi
            if self.params['modulo_theta']:
                # theta to degrees % 360
                adjusted_theta = (theta * 180 / math.pi) % 360
                # distance in degrees to north position = 0 degrees
                adjusted_theta = abs(min(abs(adjusted_theta) - 0, 360 - abs(adjusted_theta)))
                reward = (1 - (x ** 2) * self.rew_xc - (adjusted_theta ** 2) * self.rew_theta_c)
            else:
                reward = (1 - (x ** 2) * self.rew_xc - (theta ** 2) * self.rew_theta_c)
        else:
            raise ValueError(
                'cartpole parameter dictionary reward function self.params["reward_function"] should be either "original" or "variable" but is',
                self.params['reward_function'])

        self.accumulated_reward += reward
        self.accumulated_abs_theta += abs(theta)
        self.accumulated_abs_x += abs(x)
        self.num_timesteps += 1

        if self.interpolate:
            params = self.interpolate_environment_parameters()
            self.update_environment_parameters(params)
        else:
            if self.change_interval and self.num_timesteps % self.change_interval == 0:
                self.param_dict_i = (self.param_dict_i + 1) % len(self.param_dicts)
                self.params = self.param_dicts[self.param_dict_i]
                print('cartpole updating params to set number', self.param_dict_i, 'at step number', self.num_timesteps)
                self.update_environment_parameters(self.params)

        done = False
        return np.array(self.state), reward, done, {}

    def interpolate_environment_parameters(self):
        relative_steps = self.num_timesteps % self.change_interval
        cycle = (self.num_timesteps // self.change_interval) % 2
        # TODO 
        if cycle is 0:
            # w = relatiev steps / total steps
            w = relative_steps / self.change_interval
        else:
            # w = 1 - (relatiev steps / total steps)
            w = 1 - (relative_steps / self.change_interval)

        params = {}
        # for each float(i)
        for key in self.param_dicts[0]:
            params[key] = self.param_dicts[0][key]
            if type(params[key]) is float:
                params[key] = (1 - w) * self.param_dicts[0][key] + w * self.param_dicts[1][key]

        return params

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


gym.envs.register(
    id='InfHCartPole-v1',
    entry_point='cartpole:CartPoleEnv',
    max_episode_steps=float('inf'),
)
