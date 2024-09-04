import os, sys, time
from typing import Optional

import numpy as np
import yaml

import gymnasium as gym
from gymnasium import spaces
import aerialgym_arg as isaacarg

class LunarLanderI1(gym.Env):
    def __init__(self, config_path: Optional[str] = None, seed=0):
        pass

    def step(self, action):
        pass

    def reset(self, seed=None, options=None):
        pass

    def close(self):
        pass

############### private functions ####################

    def __get_reward(self, action, step):
        pass

    def __get_observation(self):
        pass

    def __get_info(self):
        pass

    def __get_initial_state(self, name):
        pass