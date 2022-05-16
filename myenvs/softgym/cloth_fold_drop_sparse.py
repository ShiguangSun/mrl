import numpy as np
import random
import pickle
import os
import os.path as osp
import pyflex
from gym import spaces
from copy import deepcopy
from softgym.utils.pyflex_utils import center_object
# from softgym.envs.cloth_fold import ClothFoldEnv
from softgym.sparse_envs.cloth_fold_sparse import ClothFoldSparseEnv


class ClothFoldDropSparseEnv(ClothFoldSparseEnv):
    def __init__(self,  reward_type = "sparse", **kwargs):
        self.start_height = 0.8
        kwargs['cached_states_path'] = 'cloth_fold_drop_init_states.pkl'
        super().__init__(**kwargs)

        self._num_key_points = 16
        self.dist_thresh = 0.05
        self.reward_type = reward_type

        # disable the patch_deprecated_methods during registration
        self._gym_disable_underscore_compat = True

        self._goal_save_dir = "save/cloth_fold_drop/goals/"
        if not osp.exists("save/cloth_fold_drop/goals/"):
            os.makedirs("save/cloth_fold_drop/goals/")

        obs = self.reset()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    def _get_drop_point_idx(self):
        return self._get_key_point_idx()[:2]

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        """pos': np.array([1.07199, 0.94942, 1.15691]"""
        config = {
            'ClothPos': [-1.6, 2.0, -0.8],
            'ClothSize': [95, 58],
            'ClothStiff': [0.9, 1.0, 0.9],  # Stretch, Bend and Shear
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': np.array([1.07199, 0.74942, 1.15691]),
                                   'angle': np.array([0.633549, -0.397932, 0]),
                                   'width': self.camera_width,
                                   'height': self.camera_height}},
            'flip_mesh': 0
        }
        return config

    def generate_env_variation(self, num_variations=1, vary_cloth_size=False):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 500  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.1  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()

        for i in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            if vary_cloth_size:
                cloth_dimx, cloth_dimy = self._sample_cloth_size()
                config['ClothSize'] = [cloth_dimx, cloth_dimy]
            else:
                cloth_dimx, cloth_dimy = config['ClothSize']
            self.set_scene(config)
            self.action_tool.reset([0., -1., 0.])

            pickpoints = self._get_drop_point_idx()[:2]  # Pick two corners of the cloth and wait until stablize

            # config['target_pos'] = self._get_flat_pos()
            self.x_low = np.random.random() * 0.2 - 0.1
            self._set_to_vertical(self.x_low, height_low=np.random.random() * 0.1 + 0.1)
            # self._set_to_vertical(x_low=np.random.random() * 0.2 - 0.1, height_low=np.random.random() * 0.1 + 0.1)

            # Get height of the cloth without the gravity. With gravity, it will be longer
            # p1, _, p2, _ = self._get_key_point_idx()
            # cloth_height = np.linalg.norm(curr_pos[p1] - curr_pos[p2])

            curr_pos = pyflex.get_positions().reshape(-1, 4)
            curr_pos[0] += np.random.random() * 0.001  # Add small jittering
            original_inv_mass = curr_pos[pickpoints, 3]
            curr_pos[pickpoints, 3] = 0  # Set mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
            pickpoint_pos = curr_pos[pickpoints, :3]
            pyflex.set_positions(curr_pos.flatten())

            picker_radius = self.action_tool.picker_radius
            self.action_tool.update_picker_boundary([-0.3, 0.05, -0.5], [0.5, 2, 0.5])
            self.action_tool.set_picker_pos(picker_pos=pickpoint_pos + np.array([0., picker_radius, 0.]))

            # Pick up the cloth and wait to stablize
            for j in range(0, max_wait_step):
                pyflex.step()
                curr_pos = pyflex.get_positions().reshape((-1, 4))
                curr_vel = pyflex.get_velocities().reshape((-1, 3))
                if np.alltrue(curr_vel < stable_vel_threshold) and j > 300:
                    break
                curr_pos[pickpoints, :3] = pickpoint_pos
                pyflex.set_positions(curr_pos)
            curr_pos = pyflex.get_positions().reshape((-1, 4))
            curr_pos[pickpoints, 3] = original_inv_mass
            pyflex.set_positions(curr_pos.flatten())
            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states

    def _reset(self):
        """ Right now only use one initial state"""
        if hasattr(self, 'action_tool'):
            particle_pos = pyflex.get_positions().reshape(-1, 4)
            drop_point_pos = particle_pos[self._get_drop_point_idx(), :3]
            middle_point = np.mean(drop_point_pos, axis=0)
            self.action_tool.reset(middle_point)  # middle point is not really useful
            picker_radius = self.action_tool.picker_radius
            self.action_tool.update_picker_boundary([-0.3, 0.5, -0.5], [0.5, 2, 0.5])
            self.action_tool.set_picker_pos(picker_pos=drop_point_pos + np.array([0., picker_radius, 0.]))

        # config = self.get_current_config()
        # num_particles = np.prod(config['ClothSize'], dtype=int)
        # particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        # cloth_dimx = config['ClothSize'][0]
        # x_split = cloth_dimx // 2
        # self.fold_group_a = particle_grid_idx[:, :x_split].flatten()
        # self.fold_group_b = np.flip(particle_grid_idx, axis=1)[:, :x_split].flatten()

        # colors = np.zeros(num_particles)
        # colors[self.fold_group_a] = 1

        # pyflex.step()
        # self.init_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        # pos_a = self.init_pos[self.fold_group_a, :]
        # pos_b = self.init_pos[self.fold_group_b, :]
        # self.prev_dist = np.mean(np.linalg.norm(pos_a - pos_b, axis=1))

        # self.performance_init = None
        # info = self._get_info()
        # self.performance_init = info['performance']
        return self._get_obs()

    def _get_current_dist(self, pos):
        goal_pos = self.get_current_config()['goal_pos']
        curr_pos = pos.reshape((-1, 4))[:, :3]
        curr_dist = np.mean(np.linalg.norm(curr_pos - goal_pos, axis=1))
        return curr_dist

    def _set_to_vertical(self, x_low, height_low):
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        vertical_pos = self._get_vertical_pos(x_low, height_low)
        curr_pos[:, :3] = vertical_pos
        max_height = np.max(curr_pos[:, 1])
        if max_height < 0.5:
            curr_pos[:, 1] += 0.5 - max_height
        pyflex.set_positions(curr_pos)
        pyflex.step()

    def _get_vertical_pos(self, x_low, height_low):
        config = self.get_current_config()
        dimx, dimy = config['ClothSize']

        x = np.array([i * self.cloth_particle_radius for i in range(dimx)])
        x = np.array(list(reversed(x)))
        y = np.array([i * self.cloth_particle_radius for i in range(dimy)])
        # x = x - np.mean(x)
        y = y - np.mean(y)
        xx, yy = np.meshgrid(x, y)

        curr_pos = np.zeros([dimx * dimy, 3], dtype=np.float32)
        curr_pos[:, 0] = x_low
        curr_pos[:, 2] = yy.flatten()
        curr_pos[:, 1] = xx.flatten() - np.min(xx) + height_low
        return curr_pos

    def _get_flat_pos(self):
        config = self.get_current_config()
        dimx, dimy = config['ClothSize']

        x = np.array([i * self.cloth_particle_radius for i in range(dimx)])
        y = np.array([i * self.cloth_particle_radius for i in range(dimy)])
        y = y - np.mean(y)
        xx, yy = np.meshgrid(x, y)
        curr_pos = np.zeros([dimx * dimy, 3], dtype=np.float32)
        curr_pos[:, 0] = xx.flatten()
        curr_pos[:, 2] = yy.flatten()
        curr_pos[:, 1] = 5e-3  # Set specifally for particle radius of 0.00625
        return curr_pos

    def _set_to_flat(self):
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        flat_pos = self._get_flat_pos()
        curr_pos[:, :3] = flat_pos
        pyflex.set_positions(curr_pos)
        pyflex.step()

    def _sample_goal(self):
        # reset scene
        self.current_config = self.cached_configs[0]
        init_state = self.cached_init_states[0]
        self.set_scene(self.current_config, init_state)

        self._set_to_flat()

        num_particles = np.prod(self.current_config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(self.current_config['ClothSize'][1], self.current_config['ClothSize'][0])  # Reversed index here

        cloth_dimx = self.current_config['ClothSize'][0]
        self.x_split = np.random.randint(cloth_dimx // 5, cloth_dimx // 2 +1)
        self.fold_group_a = particle_grid_idx[:, :self.x_split].flatten()
        self.fold_group_b = np.flip(particle_grid_idx, axis=1)[:, cloth_dimx - 2*self.x_split:cloth_dimx -self.x_split].flatten()

        colors = np.zeros(num_particles)
        colors[self.fold_group_a] = 1
      
        pyflex.step()

        curr_pos = pyflex.get_positions().reshape((-1, 4))
        curr_pos[self.fold_group_a, :] = curr_pos[self.fold_group_b, :].copy()
        curr_pos[self.fold_group_a, 1] += 0.05  # group a particle position made tcurr_pos[self.fold_group_b, 1] + 0.05e at top of group b position.
        # curr_pos[self.fold_group_a, 1] += self.cloth_particle_radius

        pyflex.set_positions(curr_pos.flatten())
        pyflex.set_velocities(np.zeros_like(curr_pos[:,:3]))
        for i in range(10):
            pyflex.step()
        center_object

        curr_pos = pyflex.get_positions().reshape((-1, 4))[:,:3]
        key_point = self._get_key_point_idx()
        goal = self._normalize_points(curr_pos[key_point,:3].flatten())
    

        # visualize the goal scene
        if hasattr(self, 'action_tool'):
            self.action_tool.reset([10, 10, 10])
        self.goal_img = self.render_goal(200, 200)
        return goal

    

    