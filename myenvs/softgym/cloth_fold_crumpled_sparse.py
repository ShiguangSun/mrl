import numpy as np
import random
import os
import os.path as osp
import pyflex
from copy import deepcopy
# from softgym.envs.cloth_fold import ClothFoldEnv
from gym import spaces
from softgym.utils.pyflex_utils import center_object
from softgym.sparse_envs.cloth_fold_sparse import ClothFoldSparseEnv

class ClothFoldCrumpledSparseEnv(ClothFoldSparseEnv):
    def __init__(self, reward_type = "sparse", **kwargs):
        kwargs['cached_states_path'] = 'cloth_fold_crumpled_init_states.pkl'
        super().__init__(**kwargs)

        self._num_key_points = 16
        self.dist_thresh = 0.05
        self.reward_type = reward_type

        # disable the patch_deprecated_methods during registration
        self._gym_disable_underscore_compat = True

        self._goal_save_dir = "save/cloth_crumpled_drop/goals/"
        if not osp.exists("save/cloth_crumpled_drop/goals/"):
            os.makedirs("save/cloth_crumpled_drop/goals/")

        obs = self.reset()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    def generate_env_variation(self, num_variations=1, vary_cloth_size=False):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 300  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.01  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()
        default_config['flip_mesh'] = 1

        for i in range(num_variations):
            config = deepcopy(default_config)
            cam_pos = np.array([0.3, 0.82, 0.82])
            config['camera_params'][config['camera_name']]['pos'] = cam_pos
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            if vary_cloth_size:
                cloth_dimx, cloth_dimy = self._sample_cloth_size()
                config['ClothSize'] = [cloth_dimx, cloth_dimy]
            else:
                cloth_dimx, cloth_dimy = config['ClothSize']
            self.set_scene(config)

            self.action_tool.reset([0., -1., 0.])
            pyflex.step()

            num_particle = cloth_dimx * cloth_dimy
            pickpoint = random.randint(0, num_particle - 1)
            curr_pos = pyflex.get_positions()
            original_inv_mass = curr_pos[pickpoint * 4 + 3]
            curr_pos[pickpoint * 4 + 3] = 0  # Set the mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
            pickpoint_pos = curr_pos[pickpoint * 4: pickpoint * 4 + 3].copy()  # Pos of the pickup point is fixed to this point
            pickpoint_pos[1] += np.random.random(1) * 0.5 + 0.5
            pyflex.set_positions(curr_pos)

            # Pick up the cloth and wait to stablize
            for _ in range(0, max_wait_step):
                pyflex.step()
                curr_pos = pyflex.get_positions()
                curr_vel = pyflex.get_velocities()
                if np.alltrue(curr_vel < stable_vel_threshold):
                    break
                curr_pos[pickpoint * 4: pickpoint * 4 + 3] = pickpoint_pos
                curr_vel[pickpoint * 3: pickpoint * 3 + 3] = [0, 0, 0]
                pyflex.set_positions(curr_pos)
                pyflex.set_velocities(curr_vel)

            # Drop the cloth and wait to stablize
            curr_pos = pyflex.get_positions()
            curr_pos[pickpoint * 4 + 3] = original_inv_mass
            pyflex.set_positions(curr_pos)
            for _ in range(max_wait_step):
                pyflex.step()
                curr_vel = pyflex.get_velocities()
                if np.alltrue(curr_vel < stable_vel_threshold):
                    break

            center_object()

            if self.action_mode == 'sphere' or self.action_mode.startswith('picker'):
                curr_pos = pyflex.get_positions()
                self.action_tool.reset(curr_pos[pickpoint * 4:pickpoint * 4 + 3] + [0., 0.2, 0.])
            generated_configs.append(deepcopy(config))
            generated_states.append(deepcopy(self.get_state()))
            self.current_config = config  # Needed in _set_to_flatten function
            print('config {}: camera params {}'.format(i, config['camera_params']))

        return generated_configs, generated_states

    def _reset(self):
        """ Right now only use one initial state"""
        if hasattr(self, 'action_tool'):
            self.action_tool.reset([0., 0.2, 0.])

        # config = self.get_current_config()
        # self.flat_pos = self._get_flat_pos()
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

    # def compute_reward(self, action=None, obs=None, set_prev_reward=False):
    #     """
    #     The particles are splitted into two groups. The reward will be the minus average eculidean distance between each
    #     particle in group a and the crresponding particle in group b
    #     :param pos: nx4 matrix (x, y, z, inv_mass)
    #     """
    #     pos = pyflex.get_positions()
    #     pos = pos.reshape((-1, 4))[:, :3]
    #     pos_group_a = pos[self.fold_group_a]
    #     pos_group_b = pos[self.fold_group_b]
    #     pos_group_b_init = self.flat_pos[self.fold_group_b]
    #     curr_dist = np.mean(np.linalg.norm(pos_group_a - pos_group_b, axis=1)) + 1.2 * np.linalg.norm(np.mean(pos_group_b - pos_group_b_init, axis=1))
    #     reward = -curr_dist
    #     return reward

    def _get_info(self):
        # Duplicate of the compute reward function!
        # pos = pyflex.get_positions()
        # pos = pos.reshape((-1, 4))[:, :3]
        # pos_group_a = pos[self.fold_group_a]
        # pos_group_b = pos[self.fold_group_b]
        # pos_group_b_init = self.init_pos[self.fold_group_b]
        # group_dist = np.mean(np.linalg.norm(pos_group_a - pos_group_b, axis=1))
        # fixation_dist = np.mean(np.linalg.norm(pos_group_b - pos_group_b_init, axis=1))
        # performance = -group_dist - 1.2 * fixation_dist
        # performance_init = performance if self.performance_init is None else self.performance_init  # Use the original performance
        # return {
        #     'performance': performance,
        #     'normalized_performance': (performance - performance_init) / (0. - performance_init),
        #     'neg_group_dist': -group_dist,
        #     'neg_fixation_dist': -fixation_dist
        # }
        return {}
