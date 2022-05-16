import numpy as np
import random
import pickle
import os
import os.path as osp
import pyflex
from gym.spaces import Box
from gym import spaces
from softgym.envs.rope_flatten import RopeFlattenEnv
from softgym.utils.pyflex_utils import random_pick_and_place, center_object
import scipy
import copy
from copy import deepcopy
import scipy.optimize as opt
import cv2

class RopeAlphabetSparseEnv(RopeFlattenEnv):
    def __init__(self, cached_states_path='rope_configuration_init_states.pkl', reward_type='sparse', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:

        manipulate the rope into a given character shape.
        """
        kwargs["num_picker"] = 1
        self.goal_characters = ['S', 'O', 'C', 'U', 'D']
        self.reward_type = reward_type
        self.dist_thresh = 0.05
        super().__init__(cached_states_path=cached_states_path, **kwargs)

        # disable the patch_deprecated_methods during registration
        self._gym_disable_underscore_compat = True

        self._goal_save_dir = "save/rope_alphabet/goals/"
        if not osp.exists("save/rope_alphabet/goals/"):
            os.makedirs("save/rope_alphabet/goals/")

        obs = self.reset()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
       
        # # change observation space: add goal character information
        # if self.observation_mode in ['key_point', 'point_cloud']:
        #     if self.observation_mode == 'key_point':
        #         obs_dim = 10 * 6 # evenly sample particles from current state and goal state
        #     else:
        #         max_particles = 160
        #         obs_dim = max_particles * 3
        #         self.particle_obs_dim = obs_dim * 2 # include both current particle position and goal particle position
        #     if self.action_mode in ['picker']:
        #         obs_dim += self.num_picker * 3
        #     else:
        #         raise NotImplementedError
        #     self.observation_space = Box(np.array([-np.inf] * obs_dim), np.array([np.inf] * obs_dim), dtype=np.float32)
        # elif self.observation_mode == 'cam_rgb':
        #     self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 3 * 2),
        #                                  dtype=np.float32) # stack current image and goal image
    def get_default_config(self, c='C'):
        """ Set the default config of the environment and load it to self.config """
        config = super().get_default_config()
        config['goal_character'] = c
        return config

    def generate_env_variation(self, num_variations=1, save_to_file=False, **kwargs):
        """
        Just call RopeFlattenEnv's generate env variation, and then add the target character's position.
        """
        self.generate_alphabet_positions()
        # self.generate_alphabet_image()
        super_config = copy.deepcopy(self.get_default_config())
        del super_config["goal_character"]

        cached_configs, cached_init_states = super().generate_env_variation(config=super_config, 
            num_variations=1, save_to_file=False)
        self.action_tool.reset([0., -1., 0.])
        
        for idx, cached_config in enumerate(cached_configs):
            goal_character = self.goal_characters[np.random.choice(len(self.goal_characters))]
            cached_config['goal_character'] = goal_character
            cached_config['goal_character_pos'] = self.goal_characters_position[goal_character]
            # cached_config['goal_character_img'] = self.goal_characters_image[goal_character]
            print("config {} GoalCharacter {}".format(idx, goal_character))

        # cached_configs, cached_init_states = [], []
        # # generated_configs, generated_states = [], []
        # # if config is None:
        # config = self.get_default_config()
        # default_config = config            
        # for i in range(num_variations):
        #     config = deepcopy(default_config)
        #     config['segment'] = self.get_random_rope_seg_num()
            
        #     self.set_scene(config)

        #     self.update_camera('default_camera', default_config['camera_params']['default_camera'])
        #     config['camera_params'] = deepcopy(self.camera_params)
        #     self.action_tool.reset([0., -1., 0.])

        #     random_pick_and_place(pick_num=4, pick_scale=0.005)
        #     center_object()
        #     cached_configs.append(deepcopy(config))
        #     print('config {}: {}'.format(i, config['camera_params']))
        #     cached_init_states.append(deepcopy(self.get_state()))

            # config = deepcopy(default_config)
            # config['segment']=60
            # config['goal_character'] = 'M'
            # self.set_scene(config)

            # self.update_camera('default_camera', default_config['camera_params']['default_camera'])
            # config['camera_params'] = deepcopy(self.camera_params)
            # self.action_tool.reset([0., -1., 0.])

            # random_pick_and_place(pick_num=4, pick_scale=0.005)
            # center_object()
            # cached_configs.append(deepcopy(config))
            # print('config {}: {}'.format(i, config['camera_params']))
            # cached_init_states.append(deepcopy(self.get_state()))

        # return generated_configs, generated_states
        # print(cached_configs)
        return cached_configs, cached_init_states
        

    def generate_alphabet_positions(self):
        self.goal_characters_position = {}
        cur_dir = osp.dirname(osp.abspath(__file__))
        character_loc_path = osp.join(cur_dir, './files', 'rope_alphabet.pkl')
        character_locs = pickle.load(open(character_loc_path, 'rb'))

        for c in character_locs:
            config = self.get_default_config(c=c)
            inv_mass = 1. / config['mass']
            radius = config['radius'] * config['scale']
            particle_num = int(config['segment'] + 1)

            pos = np.zeros((particle_num, 4))
            x, y = character_locs[c]
            if len(x) > particle_num:
                all_idxes = [x for x in range(1, len(x) - 1)]
                chosen_idxes = np.random.choice(all_idxes, particle_num - 2, replace=False)
                chosen_idxes = list(np.sort(chosen_idxes))
                chosen_idxes = [0] + chosen_idxes + [len(x) - 1]
                x = np.array(x)[chosen_idxes]
                y = np.array(y)[chosen_idxes]
            elif particle_num > len(x):
                interpolate_idx = np.random.choice(range(1, len(x) - 1), particle_num - len(x), replace=False)
                interpolate_idx = list(np.sort(interpolate_idx))
                interpolate_idx = [0] + interpolate_idx
                x_new = []
                y_new = []
                print('interpolate_idx: ', interpolate_idx)
                for idx in range(1, len(interpolate_idx)):
                    [x_new.append(x[_]) for _ in range(interpolate_idx[idx - 1], interpolate_idx[idx])]
                    [y_new.append(y[_]) for _ in range(interpolate_idx[idx - 1], interpolate_idx[idx])]
                    print(interpolate_idx[idx])
                    [print(_, end=' ') for _ in range(interpolate_idx[idx - 1], interpolate_idx[idx])]
                    x_new.append((x[interpolate_idx[idx]] + x[interpolate_idx[idx] + 1]) / 2)
                    y_new.append((y[interpolate_idx[idx]] + y[interpolate_idx[idx] + 1]) / 2)
                [x_new.append(x[_]) for _ in range(interpolate_idx[-1], len(x))]
                [y_new.append(y[_]) for _ in range(interpolate_idx[-1], len(y))]
                [print(_, end=' ') for _ in range(interpolate_idx[-1], len(y))]
                x = x_new
                y = y_new

            for p_idx in range(particle_num):
                pos[p_idx][0] = y[p_idx] * radius
                pos[p_idx][1] = 0.05 # y
                pos[p_idx][2] = x[p_idx] * radius
                pos[p_idx][3] = inv_mass

            pos[:, 0] -= np.mean(pos[:, 0])
            pos[:, 2] -= np.mean(pos[:, 2])

            self.goal_characters_position[c] = pos.copy()

    def generate_alphabet_image(self):
        self.goal_characters_image = {}
        for c in self.goal_characters:
            default_config = self.get_default_config(c=c)
            goal_c_pos =  self.goal_characters_position[c]
            self.set_scene(default_config)
            all_positions = pyflex.get_positions().reshape([-1, 4])
            all_positions = goal_c_pos.copy() # ignore the first a few cloth particles
            pyflex.set_positions(all_positions)
            self.update_camera('default_camera', default_config['camera_params']['default_camera']) # why we need to do this?
            self.action_tool.reset([0., -1., 0.]) # hide picker
            # goal_c_img = self.get_image(self.camera_height, self.camera_width)

            # import time
            # for _ in range(50):   
            #     pyflex.step(render=True)
                # time.sleep(0.1)
                # cv2.imshow('img', self.get_image())
                # cv2.waitKey()
                
            goal_c_img = self.get_image(self.camera_height, self.camera_width)
            # cv2.imwrite('../data/images/rope-configuration-goal-image-{}.png'.format(c), goal_c_img[:,:,::-1])
            # exit()

            self.goal_characters_image[c] = goal_c_img.copy()

    # def compute_reward(self, action=None, obs=None, **kwargs):
    #     """ Reward is the matching degree to the goal character"""
    #     goal_c_pos = self.current_config["goal_character_pos"][:, :3]
    #     current_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        
    #     # way1: index matching
    #     if self.reward_type == 'index':
    #         dist = np.linalg.norm(current_pos - goal_c_pos, axis=1)
    #         reward = -np.mean(dist)

    #     if self.reward_type == 'bigraph':
    #         # way2: downsample and then use Hungarian algorithm for bipartite graph  matching
    #         downsampled_cur_pos = current_pos[self.key_point_indices]
    #         downsampled_goal_pos = goal_c_pos[self.key_point_indices]
    #         W = np.zeros((len(downsampled_cur_pos), len(downsampled_cur_pos)))
    #         for idx in range(len(downsampled_cur_pos)):
    #             all_dist = np.linalg.norm(downsampled_cur_pos[idx] - downsampled_goal_pos, axis=1)
    #             W[idx, :] = all_dist
            
    #         row_idx, col_idx = opt.linear_sum_assignment(W)
    #         dist = W[row_idx, col_idx].sum()
    #         reward = -dist / len(downsampled_goal_pos)
        
    #     return reward

    def compute_reward(self, achieved_goal, desired_goal, info):
        """ Reward is the distance between the endpoints of the rope"""
        _shape = achieved_goal.shape[:-1] + (len(self.key_point_indices), 3)
        achieved_goal = achieved_goal.reshape(_shape)
        desired_goal = desired_goal.reshape(_shape)

        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1).max(-1)

        if self.reward_type == "sparse":
            return - (dist >= self.dist_thresh).astype(np.float32)
        else:
            return - dist

    def reset(self):
        self.goal = self._sample_goal()

        self.current_config = self.cached_configs[0]
        self.set_scene(self.cached_configs[0], self.cached_init_states[0])
        # if self.current_config['segment'] == 'M':
        #     self.set_scene(self.current_config, self.cached_init_states[1])
        # else:
        #     self.set_scene(self.current_config, self.cached_init_states[0])
        random_pick_and_place(pick_num=5, pick_scale=0.001)
        center_object()
        self.prev_reward = 0.
        self.time_step = 0
        obs = self._reset()
        achieved_goal = obs.reshape(-1, 3)[:len(self.key_point_indices)].flatten()

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy()
        }

    def step(self, action):
        """ If record_continuous_video is set to True, will record an image for each sub-step"""
        frames = []
        # process action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = action.reshape(-1, 4)
        action[:, 1] = - 10.
        action = action.reshape(-1)
        for i in range(self.action_repeat):
            self._step(action)

        obs = self._get_obs()
        achived_goal = obs.copy().reshape(-1, 3)[:len(self.key_point_indices)].flatten()
        achived_goal = self._normalize_points(achived_goal)
        obs = self._normalize_points(obs)

        desired_goal = self.goal
        reward = self.compute_reward(achived_goal, desired_goal, None)

        obs = {
            "observation": obs.copy(),
            "achieved_goal": achived_goal.copy(),
            "desired_goal": desired_goal.copy()
        }
        info = {'is_success': reward >= - self.dist_thresh}

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
        self.generate_alphabet_positions()
        # self.generate_alphabet_image()
        self.goal_character = self.goal_characters[np.random.choice(len(self.goal_characters))]
        self.current_config = self.cached_configs[0]
        
        init_state = self.cached_init_states[0]
        # self.set_scene(config, init_state)
        # goal_character = self.goal_characters[np.random.choice(len(self.goal_characters))]
        # if goal_character == 'M':
        #     self.current_config['segment'] = 60
        # else:
        #     self.current_config['segment'] = 40
        # if self.goal_character == 'M':
        #     self.current_config = self.cached_configs[1]
        #     init_state = self.cached_init_states[1]
        # else:
        #     self.current_config = self.cached_configs[0]
        #     init_state = self.cached_init_states[0]
        self.set_scene(self.current_config, init_state)
        # print(self.current_config['segment'])

        self.current_config['goal_character'] = self.goal_character
        self.current_config['goal_character_pos'] = self.goal_characters_position[self.goal_character]
        # self.current_config['goal_character_img'] = self.goal_characters_image[goal_character]
        # print(self.goal_character)
        # print(self.current_config['goal_character_pos'].shape)
        # print(self.goal_characters_position[self.goal_character].shape)
        rope_particle_num = self.current_config['segment'] + 1
        self.key_point_indices = self._get_key_point_idx(rope_particle_num)
        # print(rope_particle_num)

        goal_c_pos = self.current_config["goal_character_pos"][:, :3]
        keypoint_pos = goal_c_pos[self.key_point_indices]
        goal = keypoint_pos.flatten()

        pyflex.set_positions(self.goal_characters_position[self.goal_character].flatten())
        pyflex.step()

        # visualize the goal scene
        if hasattr(self, 'action_tool'):
            self.action_tool.reset([10, 10, 10])
        self.goal_img = self.render_goal(200, 200)
        # self.goal_img = self.goal_characters_image[goal_character]

        # print("GoalCharacter {}".format(goal_character))
        return goal

    def _normalize_points(self, points):
        input_shape = points.shape
        pos = pyflex.get_positions().reshape(-1, 4)
        points = points.reshape(-1 ,3)
        points[:, [0, 2]] -= np.mean(pos[:, [0, 2]], axis=0, keepdims=True)
        return points.reshape(input_shape)

    def _get_obs(self):
        if self.observation_mode == 'cam_rgb':
            obs_img = self.get_image(self.camera_height, self.camera_width)
            goal_img = self.current_config['goal_character_img']
            ret_img = np.concatenate([obs_img, goal_img], axis=2)
            return ret_img

        if self.observation_mode == 'point_cloud':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3].flatten()
            pos = np.zeros(shape=self.particle_obs_dim, dtype=np.float)
            pos[:len(particle_pos)] = particle_pos
            pos[len(particle_pos):] = self.current_config["goal_character_pos"][:, :3].flatten()
        elif self.observation_mode == 'key_point':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
            keypoint_pos = particle_pos[self.key_point_indices, :3]
            # goal_keypoint_pos = self.current_config["goal_character_pos"][self.key_point_indices, :3]
            pos = keypoint_pos.flatten()

        if self.action_mode in ['sphere', 'picker']:
            shapes = pyflex.get_shape_states()
            shapes = np.reshape(shapes, [-1, 14])
            pos = np.concatenate([pos.flatten(), shapes[:, :3].flatten()])
        return pos

    def _reset(self):
        super()._reset()

        # self.performance_init = None
        # info = self._get_info()
        # self.performance_init = info['performance']
        # pyflex.step()
        return self._get_obs()

    def _step(self, action):
        if self.action_mode.startswith('picker'):
            self.action_tool.step(action)
            pyflex.step()
        else:
            raise NotImplementedError
        return

    # def _get_info(self):
    #     goal_c_pos = self.current_config["goal_character_pos"][:, :3]
    #     current_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        
    #     # way1: index matching
    #     if self.reward_type == 'index':
    #         dist = np.linalg.norm(current_pos - goal_c_pos, axis=1)
    #         reward = -np.mean(dist)

    #     if self.reward_type == 'bigraph':
    #         # way2: downsample and then use Hungarian algorithm for bipartite graph  matching
    #         downsampled_cur_pos = current_pos[self.key_point_indices]
    #         downsampled_goal_pos = goal_c_pos[self.key_point_indices]
    #         W = np.zeros((len(downsampled_cur_pos), len(downsampled_cur_pos)))
    #         for idx in range(len(downsampled_cur_pos)):
    #             all_dist = np.linalg.norm(downsampled_cur_pos[idx] - downsampled_goal_pos, axis=1)
    #             W[idx, :] = all_dist
            
    #         row_idx, col_idx = opt.linear_sum_assignment(W)
    #         dist = W[row_idx, col_idx].sum()
    #         reward = -dist / len(downsampled_goal_pos)

    #     performance = reward
    #     performance_init =  performance if self.performance_init is None else self.performance_init  # Use the original performance

    #     return {
    #         'performance': performance,
    #         'normalized_performance': (performance - performance_init) / (self.reward_max - performance_init),
    #     }
