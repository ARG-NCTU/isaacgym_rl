import sys
import os
import pygame
import gymnasium as gym
import warnings
import numpy as np
from aerial_gym import AERIAL_GYM_DIRECTORY
from gymnasium_arg.envs.lunarlander_i1 import task_config
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnRewardThreshold
from datetime import date


warnings.filterwarnings("ignore")
policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[80,80,80])

today = date.today()
checkpoint_callback = CheckpointCallback(
  save_freq=100000,
  save_path="./logs/",
  name_prefix="dp_"+str(today),
  save_replay_buffer=True,
  save_vecnormalize=True,
)
# task = task_config()
# task.headless = False
vec_env = make_vec_env("gymnasium_arg:lunar-lander-i1", n_envs=1)
# Parallel environments
model = PPO("MlpPolicy", vec_env, verbose=1,
            policy_kwargs=policy_kwargs, 
            batch_size=2048,
            n_steps=4096,
            n_epochs=5, 
            ent_coef=0.01,
            tensorboard_log='tensorboard_log')

model.learn(total_timesteps=20_000_000, tb_log_name='tb_ppo', callback=checkpoint_callback)
model.save("ppo_lunarlander")

# del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_cartpole")

# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("human")