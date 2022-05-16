# TODO: add the following commented environments into the register
# from BaxterReacherv0 import *
# from myenvs.robosuite.robosuite import *

import copy
from .registration import register, make, registry, spec


register(
    id='FetchThrowDice-v0',
    entry_point='myenvs.fetch:FetchThrowDiceEnv',
    kwargs={},
    max_episode_steps=50,
)

for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type':reward_type,
    }

    for i in range(2, 101):
        _kwargs = copy.deepcopy(kwargs)
        _kwargs["n_bits"] = i
        register(
            id='FlipBit{}{:d}-v0'.format(suffix, i),
            entry_point='myenvs.toy:FlipBit',
            kwargs=_kwargs,
            max_episode_steps = i,
        )

    for i in range(2, 51):
        _kwargs = copy.deepcopy(kwargs)
        _kwargs["layout"] = (i, i)
        _kwargs["max_steps"] = 2 * i - 2
        register(
            id='EmptyMaze{}{:d}-v0'.format(suffix, i),
            entry_point='myenvs.toy:EmptyMaze',
            kwargs=_kwargs,
            max_episode_steps=_kwargs["max_steps"],
        )

    register(
        id='FourRoom{}-v0'.format(suffix),
        entry_point='myenvs.toy:FourRoomMaze',
        kwargs=kwargs,
        max_episode_steps=32,
    )

    register(
        id='FetchReachDiscrete{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchReachDiscrete',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPushDiscrete{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchPushDiscrete',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchSlideDiscrete{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchSlideDiscrete',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchThrow{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchThrowEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchThrowRubberBall{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchThrowRubberBallEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPickAndThrow{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchPickAndThrowEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='DragRope{}-v0'.format(suffix),
        entry_point='myenvs.ravens:DragRopeEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='SweepPile{}-v0'.format(suffix),
        entry_point='myenvs.ravens:SweepPileEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='MsPacman{}-v0'.format(suffix),
        entry_point='myenvs.atari.mspacman:MsPacman',
        kwargs=kwargs,
        max_episode_steps=26,
    )

    _rope_kwargs = {
        'observation_mode': 'key_point',
        'action_mode': 'picker',
        'num_picker': 2,
        'render': True,
        'headless': True,
        'horizon': 50,
        'action_repeat': 8,
        'render_mode': 'cloth',
        'num_variations': 1,
        'use_cached_states': False,
        'deterministic': False,
        'save_cached_states': False,
    }
    _rope_kwargs.update(kwargs)
    register(
        id='RopeConfiguration{}-v0'.format(suffix),
        entry_point='myenvs.softgym:RopeConfigurationSparseEnv',
        kwargs=_rope_kwargs,
        max_episode_steps=50,
    )

    _rope_alphabet_kwargs = {
        'observation_mode': 'key_point',
        'action_mode': 'picker',
        'num_picker': 2,
        'render': True,
        'headless': True,
        'horizon': 50,
        'action_repeat': 8,
        'render_mode': 'cloth',
        'num_variations': 1,
        'use_cached_states': False,
        'deterministic': False,
        'save_cached_states': False,
    }
    _rope_alphabet_kwargs.update(kwargs)
    register(
        id='RopeAlphabet{}-v0'.format(suffix),
        entry_point='myenvs.softgym:RopeAlphabetSparseEnv',
        kwargs=_rope_alphabet_kwargs,
        max_episode_steps=50,
    )

    _cloth_fold_kwargs = {'observation_mode': 'key_point',
                  'action_mode': 'picker',
                  'num_picker': 2,
                  'render': True,
                  'headless': True,
                  'horizon': 100,
                  'action_repeat': 8,
                  'render_mode': 'cloth',
                  'num_variations': 1,
                  'use_cached_states': False,
                  'deterministic': False,
                  'save_cached_states': False,
                  }
    _cloth_fold_kwargs.update(kwargs)
    register(
        id='ClothFold{}-v0'.format(suffix),
        entry_point='myenvs.softgym:ClothFoldSparseEnv',
        kwargs=_cloth_fold_kwargs,
        max_episode_steps=100,
    )              

    _cloth_fold_crumpled_kwargs = {'observation_mode': 'key_point',
                          'action_mode': 'picker',
                          'num_picker': 2,
                          'render': True,
                          'headless': True,
                          'horizon': 100,
                          'action_repeat': 8,
                          'render_mode': 'cloth',
                          'num_variations': 1,
                          'use_cached_states': False,
                          'deterministic': False,
                          'save_cached_states': False,
                          }
    _cloth_fold_crumpled_kwargs.update(kwargs)
    register(
        id='ClothFoldCrumpled{}-v0'.format(suffix),
        entry_point='myenvs.softgym:ClothFoldCrumpledSparseEnv',
        kwargs=_cloth_fold_crumpled_kwargs,
        max_episode_steps=100,
    )

    _cloth_fold_drop_kwargs = {'observation_mode': 'key_point',
                      'action_mode': 'picker',
                      'num_picker': 2,
                      'render': True,
                      'headless': True,
                      'horizon': 100,
                      'action_repeat': 8,
                      'render_mode': 'cloth',
                      'num_variations': 1,
                      'use_cached_states': False,
                      'deterministic': False,
                      'save_cached_states': False,
                      }
    _cloth_fold_drop_kwargs.update(kwargs)
    register(
        id='ClothFoldDrop{}-v0'.format(suffix),
        entry_point='myenvs.softgym:ClothFoldDropSparseEnv',
        kwargs=_cloth_fold_drop_kwargs,
        max_episode_steps=100,
    )

    _cloth_drop_kwargs = dict(observation_mode='key_point',
                      action_mode='picker',
                      num_picker=2,
                      render=True,
                      headless=True,
                      horizon=50,
                      action_repeat=16,
                      render_mode='cloth',
                      num_variations=1,
                      use_cached_states=False,
                      deterministic=False,
                      save_cached_states=False)
    _cloth_drop_kwargs.update(kwargs)
    register(
        id='ClothDrop{}-v0'.format(suffix),
        entry_point='myenvs.softgym:ClothDropSparseEnv',
        kwargs=_cloth_drop_kwargs,
        max_episode_steps=50,
    ) 

    _pass_water_kwargs = dict(observation_mode='key_point',
                      action_mode='direct',
                      render=True,
                      headless=True,
                      horizon=50,
                      action_repeat=8,
                      render_mode='fluid',
                      deterministic=False,
                      num_variations=1,
                      use_cached_states=False,
                      save_cached_states=False)
    _pass_water_kwargs.update(kwargs)
    register(
        id='PassWater{}-v0'.format(suffix),
        entry_point='myenvs.softgym:PassWaterSparseEnv',
        kwargs=_pass_water_kwargs,
        max_episode_steps=50,
    )

    _pour_water_amount_kwargs = {'observation_mode': 'key_point',
                        'action_mode': 'rotation_bottom',
                        'render_mode': 'fluid',
                        'action_repeat': 8,
                        'deterministic': False,
                        'render': True,
                        'headless': True,
                        'num_variations': 1,
                        'use_cached_states': False,
                        'horizon': 100,
                        'camera_name': 'default_camera',
                        'save_cached_states': False,
                        }
    _pour_water_amount_kwargs.update(kwargs)
    register(
        id='PourWaterAmount{}-v0'.format(suffix),
        entry_point='myenvs.softgym:PourWaterAmountSparseEnv',
        kwargs=_pass_water_kwargs,
        max_episode_steps=100,
    )
