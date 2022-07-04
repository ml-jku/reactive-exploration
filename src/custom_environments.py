# Copyright 2019, The Jelly Bean World Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""Collection of JBW environments for OpenAI gym."""

from __future__ import absolute_import, division, print_function

try:
    from gym.envs.registration import register as gym_register
    from gym.envs.registration import registry

    modules_loaded = True
except:
    modules_loaded = False

try:
    env_list = ['JBW-v1', 'JBW-render-v1', 'JBW-render-verbose-v1', 'JBW-v2', 'JBW-render-v2', 'JBW-render-verbose-v2']
    for name in env_list:
        if name in env:
            print("Remove {} from registry".format(env))
            del gym.envs.registration.registry.env_specs[env]
except:
    pass

from jbw.item import *
from jbw.simulator import *
from jbw.visualizer import pi


def register(**kwargs):
    try:
        gym_register(**kwargs)
    except Exception as e:
        print(str(e))


def make_config():
    # specify the item types
    items = []
    items.append(Item("banana", [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0], [0], False, 0.0,
                      intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[-2],
                      interaction_fns=[
                          [InteractionFunction.ZERO]  # parameters for interaction between item 0 and item 0
                      ]))
    return SimulatorConfig(max_steps_per_movement=1, vision_range=5,
                           allowed_movement_directions=[ActionPolicy.ALLOWED, ActionPolicy.DISALLOWED,
                                                        ActionPolicy.DISALLOWED, ActionPolicy.DISALLOWED],
                           allowed_turn_directions=[ActionPolicy.DISALLOWED, ActionPolicy.DISALLOWED,
                                                    ActionPolicy.ALLOWED, ActionPolicy.ALLOWED],
                           no_op_allowed=False, patch_size=32, mcmc_num_iter=4000, items=items,
                           agent_color=[0.0, 0.0, 1.0], agent_field_of_view=2 * pi,
                           collision_policy=MovementConflictPolicy.FIRST_COME_FIRST_SERVED, decay_param=0.4,
                           diffusion_param=0.14, deleted_item_lifetime=2000)


def make_config_2():
    # specify the item types
    items = []
    items.append(Item("banana", [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0, 0], [0, 0], False, 0.0,
                      intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[-2],
                      interaction_fns=[
                          [InteractionFunction.ZERO], [InteractionFunction.ZERO]
                          # parameters for interaction between item 0 and item 0
                      ]))
    items.append(Item("onion", [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0, 0], [0, 0], False, 0.0,
                      intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[-2],
                      interaction_fns=[
                          [InteractionFunction.ZERO], [InteractionFunction.ZERO]
                          # parameters for interaction between item 0 and item 0
                      ]))
    return SimulatorConfig(max_steps_per_movement=1, vision_range=5,
                           allowed_movement_directions=[ActionPolicy.ALLOWED, ActionPolicy.DISALLOWED,
                                                        ActionPolicy.DISALLOWED, ActionPolicy.DISALLOWED],
                           allowed_turn_directions=[ActionPolicy.DISALLOWED, ActionPolicy.DISALLOWED,
                                                    ActionPolicy.ALLOWED, ActionPolicy.ALLOWED],
                           no_op_allowed=False, patch_size=32, mcmc_num_iter=4000, items=items,
                           agent_color=[0.0, 0.0, 1.0], agent_field_of_view=2 * pi,
                           collision_policy=MovementConflictPolicy.FIRST_COME_FIRST_SERVED, decay_param=0.4,
                           diffusion_param=0.14, deleted_item_lifetime=2000)


def make_config_3():
    # specify the item types
    items = []
    items.append(Item("banana", [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0, 0, 0], [0, 0, 0], False, 0.0,
                      intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[-2],
                      interaction_fns=[
                          [InteractionFunction.ZERO], [InteractionFunction.ZERO], [InteractionFunction.ZERO]
                          # parameters for interaction between item 0 and item 0
                      ]))
    # items.append(Item("blueberry", [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0, 0, 0, 0], [0, 0, 0, 0], False, 0.0,
    #                   intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[-2],
    #                   interaction_fns=[
    #                       [InteractionFunction.ZERO], [InteractionFunction.ZERO], [InteractionFunction.ZERO], [InteractionFunction.ZERO]
    #                       # parameters for interaction between item 0 and item 0
    #                   ]))
    items.append(Item("onion", [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0, 0, 0], [0, 0, 0], False, 0.0,
                      intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[-2],
                      interaction_fns=[
                          [InteractionFunction.ZERO], [InteractionFunction.ZERO], [InteractionFunction.ZERO]
                          # parameters for interaction between item 0 and item 0
                      ]))
    # items.append(Item("key", [0, 0, 0.0], [0, 0, 0.0], [0, 0, 0, 1], [0, 0, 0, 1], False, 0.0,
    #                   intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[0],
    #                   interaction_fns=[
    #                       [InteractionFunction.ZERO], [InteractionFunction.ZERO], [InteractionFunction.ZERO], [InteractionFunction.ZERO]
    #                       # parameters for interaction between item 0 and item 0
    #                   ]))
    items.append(Item("wall", [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0, 0, 1], [0, 0, 0], True, 0.0,
                      intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[0],
                      interaction_fns=[
                          [InteractionFunction.ZERO], [InteractionFunction.ZERO],
                          [InteractionFunction.CROSS, 2.0, 3.0, 20.0, -50.0, -3.0, 1.0]
                          # [InteractionFunction.CROSS, 3.0, 4.0, 20.0, -200.0, -20.0, 1.0]
                          # parameters for interaction between item 0 and item 0
                      ]))
    return SimulatorConfig(max_steps_per_movement=1, vision_range=5,
                           allowed_movement_directions=[ActionPolicy.ALLOWED, ActionPolicy.DISALLOWED,
                                                        ActionPolicy.DISALLOWED, ActionPolicy.DISALLOWED],
                           allowed_turn_directions=[ActionPolicy.DISALLOWED, ActionPolicy.DISALLOWED,
                                                    ActionPolicy.ALLOWED, ActionPolicy.ALLOWED],
                           no_op_allowed=False, patch_size=32, mcmc_num_iter=4000, items=items,
                           agent_color=[0.0, 0.0, 1.0], agent_field_of_view=2 * pi,
                           collision_policy=MovementConflictPolicy.FIRST_COME_FIRST_SERVED, decay_param=0.4,
                           diffusion_param=0.14, deleted_item_lifetime=2000)


if modules_loaded:
    # Construct the simulator configurations.
    sim_config_1 = make_config()
    sim_config_2 = make_config_2()
    sim_config_3 = make_config_3()

    # Create a reward function.
    # 1 reward for each item of arbitrary time
    reward_fn_1 = lambda prev_items, items: sum(items) - sum(prev_items)
    # +1 reward for item 1, -1 reward for item 2
    reward_fn_2 = lambda prev_items, items: (0.0 + items[0] - prev_items[0]) - (0.0 + items[1] - prev_items[1])
    # -1 reward for item 1, +1 reward for item 2
    reward_fn_3 = lambda prev_items, items: - (0.0 + items[0] - prev_items[0]) + (0.0 + items[1] - prev_items[1])


    class cyclic_environment:
        def __init__(self, envs, steps_until_switch=0):
            self.steps = 0
            self.steps_until_switch = steps_until_switch
            self.envs = envs

        def getEnv(self):
            ret = self.envs[self.steps % 2]
            self.steps += 1
            return ret


    env_cycle = cyclic_environment(envs=[sim_config_1, sim_config_2])


    # non stationary reward cycle between fn 2 and fn 3
    class cyclic_reward_schedule:

        def __init__(self, steps_until_switch, reward_fns):
            '''
            :param steps_until_switch: until reward function switches
            :param reward_fns: list of reward functions to cycle through
            '''
            self.steps = 0
            self.steps_until_switch = steps_until_switch
            self.reward_fns = reward_fns
            self.fni = 0

        def reward_verbose(self, previous_items, items):
            self.steps += 1
            print('steps:', self.steps)
            ret = self.reward_fns[self.fni](previous_items, items)
            print('ret:', ret)

            if self.steps % self.steps_until_switch == 0:
                self.fni = (self.fni + 1) % len(self.reward_fns)
                print('switching to reward function', self.fni)

            return ret

        def reward(self, previous_items, items):
            self.steps += 1
            ret = self.reward_fns[self.fni](previous_items, items)

            if self.steps % self.steps_until_switch == 0:
                self.fni = (self.fni + 1) % len(self.reward_fns)

            return ret


    # non stationary reward cycle between fn 2 and fn 3
    class continuous_reward_schedule:

        def __init__(self, steps_until_switch, reward_fn):
            '''
            :param steps_until_switch: until reward function switches to negative
            :param reward_fn: function which slowly gets inverted and back, and so on
            '''
            self.steps = 0
            self.weight = 1.0
            self.steps_until_switch = steps_until_switch
            self.reward_fn = reward_fn

        def reward_verbose(self, previous_items, items):
            self.steps += 1
            t = self.steps - 1
            T = self.steps_until_switch * 2

            if t % T < T / 2:
                self.weight = 2 * (((T / 2) - (t % T)) / (T / 2) - (0.5))
            else:
                self.weight = 2 * (((t % T) - (T / 2)) / (T / 2) - (0.5))

            ret = self.weight * self.reward_fn(previous_items, items)
            print('steps: {steps}, weight: {weight}, reward: {reward}'.format(steps=self.steps, weight=self.weight,
                                                                              reward=ret))

            return ret

        def reward(self, previous_items, items):
            self.steps += 1
            t = self.steps - 1
            T = self.steps_until_switch * 2

            if t % T < T / 2:
                self.weight = 2 * (((T / 2) - (t % T)) / (T / 2) - (0.5))
            else:
                self.weight = 2 * (((t % T) - (T / 2)) / (T / 2) - (0.5))

            ret = self.weight * self.reward_fn(previous_items, items)

            return ret


    schedule_1_1e6 = cyclic_reward_schedule(1e6, [reward_fn_2, reward_fn_3])
    schedule_1_5e5 = cyclic_reward_schedule(5e5, [reward_fn_2, reward_fn_3])
    schedule_1_1e5 = cyclic_reward_schedule(1e5, [reward_fn_2, reward_fn_3])
    continuous_1_1e6 = continuous_reward_schedule(1e6, reward_fn_2)
    continuous_1_5e5 = continuous_reward_schedule(5e5, reward_fn_2)
    continuous_1_1e5 = continuous_reward_schedule(1e5, reward_fn_2)


    def verbose_reward_fn_1(prev_items, items):
        print('prev_items', prev_items)
        print('items:', items)
        print('prev sum', sum(prev_items))
        print('sum', sum((items)))
        return sum(items) - sum(prev_items)


    def verbose_reward_fn_2(prev_items, items):
        print('prev_items', prev_items)
        print('items:', items)
        ret = float((items[0] - prev_items[0]) - (items[1] - prev_items[1]))
        print('reward:', ret)
        return ret


    # register firste environment with only positive reward bananas

    register(
        id='JBW-v1',
        entry_point='jbw.environment:JBWEnv',
        kwargs={
            'sim_config': sim_config_1,
            'reward_fn': reward_fn_1,
            'render': False})

    register(
        id='JBW-render-v1',
        entry_point='jbw.environment:JBWEnv',
        kwargs={
            'sim_config': sim_config_1,
            'reward_fn': reward_fn_1,
            'render': True})

    register(
        id='JBW-render-verbose-v1',
        entry_point='jbw.environment:JBWEnv',
        kwargs={
            'sim_config': sim_config_1,
            'reward_fn': verbose_reward_fn_1,
            'render': True})

    # register second environment with pos reward bananas and negative reward onions

    register(
        id='JBW-v2',
        entry_point='jbw.environment:JBWEnv',
        kwargs={
            'sim_config': sim_config_2,
            'reward_fn': reward_fn_2,
            'render': False})

    register(
        id='JBW-render-v2',
        entry_point='jbw.environment:JBWEnv',
        kwargs={
            'sim_config': sim_config_2,
            'reward_fn': reward_fn_2,
            'render': True})

    register(
        id='JBW-render-verbose-v2',
        entry_point='jbw.environment:JBWEnv',
        kwargs={
            'sim_config': sim_config_2,
            'reward_fn': verbose_reward_fn_2,
            'render': True})

    # register second environment with cyclic non-stationary reward. 2 items with +1/-1 reward cycling back and forth

    register(
        id='JBW-cycle-1e6-v3',
        entry_point='jbw.environment:JBWEnv',
        kwargs={
            'sim_config': sim_config_2,
            'reward_fn': schedule_1_1e6.reward,
            'render': False})

    register(
        id='JBW-render-verbose-cycle-1e6-v3',
        entry_point='jbw.environment:JBWEnv',
        kwargs={
            'sim_config': sim_config_2,
            'reward_fn': schedule_1_1e6.reward,
            'render': True})

    register(
        id='JBW-cycle-5e5-v3',
        entry_point='jbw.environment:JBWEnv',
        kwargs={
            'sim_config': sim_config_2,
            'reward_fn': schedule_1_5e5.reward,
            'render': False})

    register(
        id='JBW-render-verbose-cycle-5e5-v3',
        entry_point='jbw.environment:JBWEnv',
        kwargs={
            'sim_config': sim_config_2,
            'reward_fn': schedule_1_5e5.reward,
            'render': True})

    register(
        id='JBW-cycle-1e5-v3',
        entry_point='jbw.environment:JBWEnv',
        kwargs={
            'sim_config': sim_config_2,
            'reward_fn': schedule_1_1e5.reward,
            'render': False})

    register(
        id='JBW-render-verbose-cycle-1e5-v3',
        entry_point='jbw.environment:JBWEnv',
        kwargs={
            'sim_config': sim_config_2,
            'reward_fn': schedule_1_1e5.reward,
            'render': True})

    # register second environment with slowly / continuously non-stationary reward. 2 items with +1/-1 reward (which gets inverted and back etc)

    register(
        id='JBW-continuous-1e6-v4',
        entry_point='jbw.environment:JBWEnv',
        kwargs={
            'sim_config': sim_config_2,
            'reward_fn': continuous_1_1e6.reward,
            'render': False})

    register(
        id='JBW-render-verbose-continuous-1e6-v4',
        entry_point='jbw.environment:JBWEnv',
        kwargs={
            'sim_config': sim_config_2,
            'reward_fn': continuous_1_1e6.reward_verbose,
            'render': True})

    register(
        id='JBW-continuous-5e5-v4',
        entry_point='jbw.environment:JBWEnv',
        kwargs={
            'sim_config': sim_config_2,
            'reward_fn': continuous_1_5e5.reward,
            'render': False})

    register(
        id='JBW-render-verbose-continuous-5e5-v4',
        entry_point='jbw.environment:JBWEnv',
        kwargs={
            'sim_config': sim_config_2,
            'reward_fn': continuous_1_5e5.reward_verbose,
            'render': True})

    register(
        id='JBW-continuous-1e5-v4',
        entry_point='jbw.environment:JBWEnv',
        kwargs={
            'sim_config': sim_config_2,
            'reward_fn': continuous_1_1e5.reward,
            'render': False})

    register(
        id='JBW-render-verbose-continuous-1e5-v4',
        entry_point='jbw.environment:JBWEnv',
        kwargs={
            'sim_config': sim_config_2,
            'reward_fn': continuous_1_1e5.reward_verbose,
            'render': True})

    # register second environment with 3 item types

    register(
        id='JBW-v5',
        entry_point='jbw.environment:JBWEnv',
        kwargs={
            'sim_config': sim_config_3,
            'reward_fn': reward_fn_2,
            'render': False})

    register(
        id='JBW-render-v5',
        entry_point='jbw.environment:JBWEnv',
        kwargs={
            'sim_config': sim_config_3,
            'reward_fn': reward_fn_2,
            'render': True})

    register(
        id='JBW-render-verbose-v5',
        entry_point='jbw.environment:JBWEnv',
        kwargs={
            'sim_config': sim_config_3,
            'reward_fn': verbose_reward_fn_2,
            'render': True})
