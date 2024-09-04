import os, sys, time

from typing import Optional
import numpy as np
import yaml
import gymnasium as gym
from gymnasium import spaces
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.registry.sim_registry import sim_config_registry
from aerial_gym.registry.env_registry import env_config_registry
import aerialgym_arg


FPS = 50
SCALE = 30.0 

class LunarLanderI1(gym.Env):

    metadata = {
                "render_modes": ["rgb_array", "human"],
                "render_fps": FPS,
                }

    def __init__(self,
                 seed = 0,
                 render_mode: Optional[str] = None,
                 ):

        ############ gymnasium ############
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_len,), dtype=np.float32, seed=seed)
        self.observation_space = gym.spaces.Dict({
            'uav_action': gym.spaces.Box(low=-1, high=1, shape=(self.uav_act_hist_len, self.action_len), dtype=np.float32, seed=seed),
            'uav_gazebo_pose': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.uav_act_hist_len, 4), dtype=np.float32, seed=seed),
        })

        self.counter_step = 0
        self.counter_total_step = 0
        self.counter_episode = 0
        self.episode_reward = 0.0
        self.uav_act = None
        self.prev_r = 0.0
        self.prev_h = 0.0
        self.land_cnt = 0
        self.render_mode = render_mode
        ############ gymnasium ############

        ############ isaacgym ############
        logger.debug("this is how a debug message looks like")
        logger.info("this is how an info message looks like")
        logger.warning("this is how a warning message looks like")
        logger.error("this is how an error message looks like")
        logger.critical("this is how a critical message looks like")
        self.gym_env = SimBuilder().build_env(
            sim_name="lunar_phyx",
            env_name="lunar_env",  # empty_env
            robot_name="base_quadrotor",  # "base_octarotor"
            controller_name="lee_acceleration_control",
            args=None,
            num_envs=1,
            device="cuda:0",
            headless=False if render_mode == "human" else True,
            use_warp=True,  # safer to use warp as it disables the camera when no object is in the environment
        )
        logger.info(
            "\n\n\n\n\n\n This script provides an example of a robot with constant forward acceleration directly input to the environment. \n\n\n\n\n\n"
        )
        ############ isaacgym ############

    def step(self, action):
        self.gym_env.step(action)

    def render(self):
        if self.render_mode == "human":
            self.gym_env.render()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.gym_env.reset()
        self.render()
        return self.__get_observation(), self.__get_info()

    def close(self):
        del self.gym_env
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.gym_env = None

############### private functions ####################

    def __get_reward(self, action, step):
        pass

    def __get_observation(self):
        pass

    def __get_info(self):
        pass

    def __get_initial_state(self, name):
        pass