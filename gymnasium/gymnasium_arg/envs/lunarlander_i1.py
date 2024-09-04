from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from aerial_gym.utils.logging import CustomLogger
import os, sys, time
from abc import ABC, abstractmethod


logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import aerialgym_arg
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.examples.dce_rl_navigation.dce_navigation_task import DCE_RL_Navigation_Task
from aerial_gym.utils.vae.vae_image_encoder import VAEImageEncoder
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import *

logger = CustomLogger("LunarLanderI1")

def dict_to_class(dict):
    return type("ClassFromDict", (object,), dict)

class task_config:
    seed = 1
    sim_name = "lunarlander_phyx"
    env_name = "lunarlander_env"
    robot_name = "base_quadrotor"
    controller_name = "lee_attitude_control"
    args = {}
    num_envs = 16
    use_warp = False
    headless = False
    device = "cuda:0"
    observation_space_dim = 13
    privileged_observation_space_dim = 0
    action_space_dim = 4
    episode_len_steps = 500  # real physics time for simulation is this value multiplied by sim.dt
    return_state_before_reset = False
    reward_parameters = {
        "pos_error_gain1": [2.0, 2.0, 2.0],
        "pos_error_exp1": [1 / 3.5, 1 / 3.5, 1 / 3.5],
        "pos_error_gain2": [2.0, 2.0, 2.0],
        "pos_error_exp2": [2.0, 2.0, 2.0],
        "dist_reward_coefficient": 7.5,
        "max_dist": 15.0,
        "action_diff_penalty_gain": [1.0, 1.0, 1.0],
        "absolute_action_reward_gain": [2.0, 2.0, 2.0],
        "crash_penalty": -100,
    }

class BaseTask(ABC):
    def __init__(self, task_config):
        self.task_config = task_config
        self.action_space = None
        self.observation_space = None
        self.reward_range = None
        self.metadata = None
        self.spec = None

        seed = task_config.seed
        if seed == -1:
            seed = time.time_ns() % (2**32)
        self.seed(seed)

    @abstractmethod
    def render(self, mode="human"):
        raise NotImplementedError

    def seed(self, seed):
        if seed is None or seed < 0:
            logger.info(f"Seed is not valid. Will be sampled from system time.")
            seed = time.time_ns() % (2**32)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

        logger.info("Setting seed: {}".format(seed))

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def reset_idx(self, env_ids):
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

class LunarLanderI1(gym.Env):

    def __init__(
            self, 
            task_config,
            seed=None, 
            num_envs=None, 
            headless=None, 
            device=None, 
            use_warp=None
            ):
        # overwrite the params if user has provided them
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp

        self.task_config = task_config
        self.reward_range = None
        self.metadata = None
        self.spec = None

        seed = task_config.seed
        if seed == -1:
            seed = time.time_ns() % (2**32)
        self.seed(seed)

        self.device = task_config.device
        # set the each of the elements of reward parameter to a torch tensor
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )
        logger.info("Building environment for navigation task.")
        logger.info(
            "Sim Name: {}, Env Name: {}, Robot Name: {}, Controller Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
            )
        )
        ############ isaacgym ############
        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )
        ############ isaacgym ############

        ############ gymnasium ############
        self.action_space = gym.space.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_transformation_function = self.task_config.action_transformation_function
        self.observation_space = gym.space.Dict(
            {
                "observations": gym.space.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.task_config.observation_space_dim,),
                    dtype=np.float32,
                )
            }
        )

        self.counter_step = 0
        self.counter_total_step = 0
        self.counter_episode = 0
        self.episode_reward = 0.0
        ############ gymnasium ############


    def step(self, action):
        self.counter_step += 1
        self.counter_total_step += 1
        self.sim_env.step(action)
        state = self.__get_observation()
        reward, terminated, truncated = self.__get_reward(action, self.counter_step)
        info = self.__get_info()
        return state, reward, terminated, truncated, info

    def render(self):
        return self.sim_env.render()

    def reset(self, seed=None, options=None):
        self.reset_idx(torch.arange(self.sim_env.num_envs))
        return self.__get_return_tuple()

    def reset_idx(self, env_ids):
        target_ratio = torch_rand_float_tensor(self.target_min_ratio, self.target_max_ratio)
        self.target_position[env_ids] = torch_interpolate_ratio(
            min=self.obs_dict["env_bounds_min"][env_ids],
            max=self.obs_dict["env_bounds_max"][env_ids],
            ratio=target_ratio[env_ids],
        )
        # logger.warning(f"reset envs: {env_ids}")
        self.infos = {}
        return

    def seed(self, seed):
        if seed is None or seed < 0:
            logger.info(f"Seed is not valid. Will be sampled from system time.")
            seed = time.time_ns() % (2**32)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

        logger.info("Setting seed: {}".format(seed))

    def close(self):
        self.sim_env.delete_env()

############### private functions ####################

    def __get_reward(self):
        return self.sim_env.get_obs()['reward'], False, self.sim_env.get_obs()['truncations']
    
    def __get_observation(self):
        return {'robot_state': self.sim_env.get_obs()['robot_state_tensor'],
                }

    def __get_info(self):
        return {}
    
    def __get_return_tuple(self):
        self.__process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )
    
    def __process_obs_for_task(self):
        self.task_obs["observations"][:, 0:3] = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"],
            (self.target_position - self.obs_dict["robot_position"]),
        )
        self.task_obs["observations"][:, 3:7] = self.obs_dict["robot_vehicle_orientation"]
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
        self.task_obs["observations"][:, 13:17] = self.obs_dict["robot_actions"]
        self.task_obs["observations"][:, 17:] = self.image_latents
        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

    def __compute_rewards_and_crashes(self, obs_dict):
        robot_position = obs_dict["robot_position"]
        target_position = self.target_position
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        robot_orientation = obs_dict["robot_orientation"]
        target_orientation = torch.zeros_like(robot_orientation, device=self.device)
        target_orientation[:, 3] = 1.0
        self.pos_error_vehicle_frame_prev[:] = self.pos_error_vehicle_frame
        self.pos_error_vehicle_frame[:] = quat_rotate_inverse(
            robot_vehicle_orientation, (target_position - robot_position)
        )
        return compute_reward(
            self.pos_error_vehicle_frame,
            self.pos_error_vehicle_frame_prev,
            obs_dict["crashes"],
            obs_dict["robot_actions"],
            obs_dict["robot_prev_actions"],
            self.curriculum_progress_fraction,
            self.task_config.reward_parameters,
        )
    
@torch.jit.script
def compute_reward(
    pos_error,
    prev_pos_error,
    crashes,
    action,
    prev_action,
    curriculum_progress_fraction,
    parameter_dict,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, Dict[str, Tensor]) -> Tuple[Tensor, Tensor]
    MULTIPLICATION_FACTOR_REWARD = (1.0 + (2.0) * curriculum_progress_fraction) * 3.0
    dist = torch.norm(pos_error, dim=1)
    prev_dist_to_goal = torch.norm(prev_pos_error, dim=1)
    pos_reward = exponential_reward_function(
        parameter_dict["pos_reward_magnitude"],
        parameter_dict["pos_reward_exponent"],
        dist,
    )
    very_close_to_goal_reward = exponential_reward_function(
        parameter_dict["very_close_to_goal_reward_magnitude"],
        parameter_dict["very_close_to_goal_reward_exponent"],
        dist,
    )
    getting_closer_reward = parameter_dict["getting_closer_reward_multiplier"] * (
        prev_dist_to_goal - dist
    )
    distance_from_goal_reward = (20.0 - dist) / 20.0
    action_diff = action - prev_action
    x_diff_penalty = exponential_penalty_function(
        parameter_dict["x_action_diff_penalty_magnitude"],
        parameter_dict["x_action_diff_penalty_exponent"],
        action_diff[:, 0],
    )
    z_diff_penalty = exponential_penalty_function(
        parameter_dict["z_action_diff_penalty_magnitude"],
        parameter_dict["z_action_diff_penalty_exponent"],
        action_diff[:, 2],
    )
    yawrate_diff_penalty = exponential_penalty_function(
        parameter_dict["yawrate_action_diff_penalty_magnitude"],
        parameter_dict["yawrate_action_diff_penalty_exponent"],
        action_diff[:, 3],
    )
    action_diff_penalty = x_diff_penalty + z_diff_penalty + yawrate_diff_penalty
    # absolute action penalty
    x_absolute_penalty = curriculum_progress_fraction * exponential_penalty_function(
        parameter_dict["x_absolute_action_penalty_magnitude"],
        parameter_dict["x_absolute_action_penalty_exponent"],
        action[:, 0],
    )
    z_absolute_penalty = curriculum_progress_fraction * exponential_penalty_function(
        parameter_dict["z_absolute_action_penalty_magnitude"],
        parameter_dict["z_absolute_action_penalty_exponent"],
        action[:, 2],
    )
    yawrate_absolute_penalty = curriculum_progress_fraction * exponential_penalty_function(
        parameter_dict["yawrate_absolute_action_penalty_magnitude"],
        parameter_dict["yawrate_absolute_action_penalty_exponent"],
        action[:, 3],
    )
    absolute_action_penalty = x_absolute_penalty + z_absolute_penalty + yawrate_absolute_penalty
    total_action_penalty = action_diff_penalty + absolute_action_penalty

    # combined reward
    reward = (
        MULTIPLICATION_FACTOR_REWARD
        * (
            pos_reward
            + very_close_to_goal_reward
            + getting_closer_reward
            + distance_from_goal_reward
        )
        + total_action_penalty
    )

    reward[:] = torch.where(
        crashes > 0,
        parameter_dict["collision_penalty"] * torch.ones_like(reward),
        reward,
    )
    return reward, crashes