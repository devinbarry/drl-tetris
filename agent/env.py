from ray.tune.registry import register_env
from nes_py.wrappers import JoypadSpace
from ray.rllib.env.atari_wrappers import WarpFrame

import gym_tetris
from gym_tetris.actions import MOVEMENT


def tetris_env_creator(version="TetrisA-v0"):
    def env_creator(env_config):
        env = gym_tetris.make(version)
        env = JoypadSpace(env, MOVEMENT)
        env = WarpFrame(env, dim=84)
        return env
    return env_creator


register_env("TetrisA-v0", tetris_env_creator("TetrisA-v0"))
register_env("TetrisA-v1", tetris_env_creator("TetrisA-v1"))
register_env("TetrisA-v2", tetris_env_creator("TetrisA-v2"))
register_env("TetrisA-v3", tetris_env_creator("TetrisA-v3"))
