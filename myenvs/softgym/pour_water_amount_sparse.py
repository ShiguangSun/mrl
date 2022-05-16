import numpy as np
from gym.spaces import Box
import pickle
import os
import os.path as osp
import cv2

import pyflex
from softgym.envs.pour_water import PourWaterPosControlEnv
import copy
from softgym.utils.misc import quatFromAxisAngle
import pickle
from gym import spaces
import time
import datetime


class PourWaterAmountPosControlSparseEnv(PourWaterPosControlEnv):
    def __init__(self, observation_mode, action_mode, reward_type = "sparse",
                config=None, cached_states_path='pour_water_amount_init_states.pkl', **kwargs):
        '''
        This class implements a pouring water task. Pouring a specific amount of water into the target cup.
        
        observation_mode: "cam_rgb" or "point_cloud"
        action_mode: "direct"
        
        '''

        super().__init__(observation_mode, action_mode, config, cached_states_path, **kwargs)
        
        self.water_thresh = 0.05
        self.reward_type = reward_type

        # disable the patch_deprecated_methods during registration
        self._gym_disable_underscore_compat = True

        # override the observation/state space to include the target amount
        if observation_mode in ['point_cloud', 'key_point']:
            if observation_mode == 'key_point':
                obs_dim = 13
                # z and theta of the second cup (poured_glass) does not change and thus are omitted.
                # Pos (x, y, z, theta) and shape (w, h, l) of the two cups, the water height and target amount
            else:
                max_particle_num = 13 * 13 * 13 * 3
                obs_dim = max_particle_num * 3
                self.particle_obs_dim = obs_dim
                obs_dim += 12 # Pos (x, y, z, theta) and shape (w, h, l) of the two cups, and target amount
            
            self.observation_space = Box(low=np.array([-np.inf] * obs_dim), high=np.array([np.inf] * obs_dim), dtype=np.float32)
        elif observation_mode == 'cam_rgb':
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 3 + 1),
                                         dtype=np.float32)

        self._goal_save_dir = "save/pour_water_amount/goals/"
        if not osp.exists("save/pour_water_amount/goals/"):
            os.makedirs("save/pour_water_amount/goals/")

        obs = self.reset()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    def get_default_config(self):
        config = super().get_default_config()
        config['target_height'] = 0.8
        return config

    def get_state(self):
        '''
        get the postion, velocity of flex particles, and postions of flex shapes.
        '''
        particle_pos = pyflex.get_positions()
        particle_vel = pyflex.get_velocities()
        shape_position = pyflex.get_shape_states()
        return {'particle_pos': particle_pos, 'particle_vel': particle_vel, 'shape_pos': shape_position,
                'glass_x': self.glass_x, 'glass_y': self.glass_y, 'glass_rotation': self.glass_rotation,
                'glass_states': self.glass_states, 'poured_glass_states': self.poured_glass_states,
                'glass_params': self.glass_params, 'config_id': self.current_config_id, 
                'line_box_x': self.line_box_x, 'line_box_y': self.line_box_y}

    def set_state(self, state_dic):
        self.line_box_x = state_dic['line_box_x']
        self.line_box_y = state_dic['line_box_y']
        super().set_state(state_dic)

    def set_shape_states(self, glass_states, poured_glass_states):
        all_states = np.concatenate((glass_states, poured_glass_states), axis=0)

        if self.line_box_x is not None:
            quat = quatFromAxisAngle([0, 0, -1.], 0.)
            indicator_box_line_states = np.zeros((1, self.dim_shape_state))

            indicator_box_line_states[0, :3] = np.array([self.line_box_x, self.line_box_y, 0.])
            indicator_box_line_states[0, 3:6] = np.array([self.line_box_x, self.line_box_y, 0.])
            indicator_box_line_states[:, 6:10] = quat
            indicator_box_line_states[:, 10:] = quat

            all_states = np.concatenate((all_states, indicator_box_line_states), axis=0)
        
        pyflex.set_shape_states(all_states)
        if self.line_box_x is not None:
            pyflex.step(render=True)
            # time.sleep(20)

    def set_scene(self, config, states=None):
        self.line_box_x = self.line_box_y = None

        if states is None:
            super().set_scene(config=config, states=states, create_only=False) # this adds the water, the controlled cup, and the target cup.
        else:
            super().set_scene(config=config, states=states, create_only=True) # this adds the water, the controlled cup, and the target cup.

        # needs to add an indicator box for how much water we want to pour into the target cup
        # create an line on the left wall of the target cup to indicate how much water we want to pour into it.
        halfEdge = np.array([0.005 / 2., 0.005 / 2., (self.poured_glass_dis_z - 2 * self.poured_border) / 2.])
        center = np.array([0., 0., 0.])
        quat = quatFromAxisAngle([0, 0, -1.], 0.)
        # print("pourwater amount add trigger box")
        pyflex.add_box(halfEdge, center, quat, 1) # set trigger to be true to create a different color for the indicator box.

        max_water_height = self._get_current_water_height() - self.border / 2
        controlled_size = (self.glass_dis_x) * (self.glass_dis_z)
        target_size = (self.poured_glass_dis_x) * (self.poured_glass_dis_z)
        estimated_target_water_height = max_water_height * config['target_amount'] / (target_size / controlled_size)
        self.line_box_x = self.x_center + self.glass_distance - self.poured_glass_dis_x / 2
        self.line_box_y = self.poured_border * 0.5 + estimated_target_water_height


        if states is None:
            self.set_shape_states(self.glass_states, self.poured_glass_states)
        else:
            self.set_state(states)

    def generate_env_variation(self, num_variations=5, **kwargs):
        """
        Just call PourWater1DPosControl's generate env variation, and then add the target amount.
        """
        config = self.get_default_config()
        super_config = copy.deepcopy(config)
        super_config['target_amount'] = np.random.uniform(0.2, 1)
        cached_configs, cached_init_states = super().generate_env_variation(config=super_config, num_variations=self.num_variations)
        return cached_configs, cached_init_states

    def _get_obs(self):
        '''
        return the observation based on the current flex state.
        '''
        if self.observation_mode == 'cam_rgb':
            return self.get_image(self.camera_width, self.camera_height)

        elif self.observation_mode in ['point_cloud', 'key_point']:
            state_dic = self.get_state()
            water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
            water_num = len(water_state)
            in_poured_glass = self.in_glass(water_state, self.poured_glass_states, self.poured_border, self.poured_height)
            in_control_glass = self.in_glass(water_state, self.glass_states, self.border, self.height)
        
            good_water = in_poured_glass * (1 - in_control_glass) # prevent to move the controlled cup directly into the target cup
            good_water_num = np.sum(good_water)

            if self.observation_mode == 'point_cloud':
                particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3].flatten()
                pos = np.zeros(shape=self.particle_obs_dim, dtype=np.float)
                pos[:len(particle_pos)] = particle_pos
                cup_state = np.array([self.glass_x, self.glass_y, self.glass_rotation, self.glass_dis_x, self.glass_dis_z, self.height,
                                  self.glass_distance + self.glass_x, self.poured_height, self.poured_glass_dis_x, self.poured_glass_dis_z,
                                  self.line_box_y, good_water_num])
            else:
                pos = np.empty(0, dtype=np.float)

                cup_state = np.array([self.glass_x, self.glass_y, self.glass_rotation, self.glass_dis_x, self.glass_dis_z, self.height,
                                  self.glass_distance + self.glass_x, self.poured_height, self.poured_glass_dis_x, self.poured_glass_dis_z,
                                  self._get_current_water_height(), self.line_box_y, good_water_num])
            
            return np.hstack([pos, cup_state]).flatten()
        else:
            raise NotImplementedError

    # def compute_reward(self, obs=None, action=None, **kwargs):
    #     """
    #     The reward is computed as the fraction of water in the poured glass.
    #     NOTE: the obs and action params are made here to be compatiable with the MultiTask env wrapper.
    #     """
    #     state_dic = self.get_state()
    #     water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
    #     water_num = len(water_state)

    #     in_poured_glass = self.in_glass(water_state, self.poured_glass_states, self.poured_border, self.poured_height)
    #     in_control_glass = self.in_glass(water_state, self.glass_states, self.border, self.height)
        
    #     good_water = in_poured_glass * (1 - in_control_glass) # prevent to move the controlled cup directly into the target cup
    #     good_water_num = np.sum(good_water)
    #     target_water_num = int(water_num * self.current_config['target_amount'])
    #     diff = np.abs(target_water_num - good_water_num) / water_num

    #     reward = - diff
    #     return reward

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        The reward is computed as the fraction of water in the poured glass.
        NOTE: the obs and action params are made here to be compatiable with the MultiTask env wrapper.
        """
        good_water_num = achieved_goal
        target_water_num = desired_goal
        diff = np.abs(target_water_num - good_water_num) / target_water_num

        if self.reward_type == "sparse":
            return -(diff >= self.water_thresh).astype(np.float32)
        else:
            return - diff

    def reset(self):
        self.goal = self._sample_goal()

        self.current_config = self.cached_configs[0]
        self.set_scene(self.cached_configs[0], self.cached_init_states[0])
        self.particle_num = pyflex.get_n_particles()
        self.prev_reward = 0.
        self.time_step = 0
        obs = self._reset()
        achieved_goal = obs[-1]

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal
        }

    def step(self, action):
        """ If record_continuous_video is set to True, will record an image for each sub-step"""
        frames = []
        # process action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        for i in range(self.action_repeat):
            self._step(action)

        obs = self._get_obs()
        achieved_goal = obs[-1]

        desired_goal = self.goal
        reward = self.compute_reward(achieved_goal, desired_goal, None)

        obs = {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": desired_goal
        }
        info = {'is_success': reward >= - self.water_thresh}

        self.time_step += 1

        done = False
        if self.time_step >= self.horizon:
            done = True
        return obs, reward, done, info

    def render_goal(self, target_w, target_h):
        img = pyflex.render()
        width, height = self.camera_params['default_camera']['width'], self.camera_params['default_camera']['height']
        img = img.reshape(height, width, 4)[::-1, :, :3]  # Need to reverse the height dimension
        img = img[int(0.25 * height):int(0.75 * height), int(0.25 * width):int(0.75 * width)]
        img = cv2.resize(img.astype(np.uint8), (target_w, target_h))
        return img

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            img = pyflex.render()
            width, height = self.camera_params['default_camera']['width'], self.camera_params['default_camera']['height']
            img = img.reshape(height, width, 4)[::-1, :, :3]  # Need to reverse the height dimension
            goal_img = self.goal_img.copy()
            # attach goal patch on the rendered image
            goal_img[:10, :, :] = 0
            goal_img[:, :10, :] = 0
            goal_img[-10:, :, :] = 0
            goal_img[:, -10:, :] = 0
            img[30:230, 30:230] = goal_img
            return img
        elif mode == 'human':
            raise NotImplementedError

    def _sample_goal(self):
        # reset scene
        self.cached_configs[0]['target_amount'] = np.random.uniform(0.2, 1)
        config = self.cached_configs[0]
        init_state = self.cached_init_states[0]
        self.set_scene(config, init_state)

        state_dic = self.get_state()
        water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
        water_num = len(water_state)

        goal = int(self.cached_configs[0]['target_amount']*water_num)

        # visualize the goal scene
        if hasattr(self, 'action_tool'):
            self.action_tool.reset([10, 10, 10])
        self.goal_img = self.render_goal(200, 200)
        return goal

    def _get_info(self):
        # Duplicate of the compute reward function!
        state_dic = self.get_state()
        water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
        water_num = len(water_state)

        in_poured_glass = self.in_glass(water_state, self.poured_glass_states, self.poured_border, self.poured_height)
        in_control_glass = self.in_glass(water_state, self.glass_states, self.border, self.height)
        good_water = in_poured_glass * (1 - in_control_glass)
        good_water_num = np.sum(good_water)
        target_water_num = int(water_num * self.current_config['target_amount'])
        diff = np.abs(target_water_num - good_water_num) / target_water_num

        reward = - diff 

        performance = reward
        performance_init =  performance if self.performance_init is None else self.performance_init  # Use the original performance

        return {
            'normalized_performance': (performance - performance_init) / (self.reward_max - performance_init),
            'performance': performance,
            'target': self.current_config['target_amount']
        }