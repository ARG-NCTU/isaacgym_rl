import sys, time
import os

import pygame
import gymnasium as gym
from gymnasium.utils.play import play, display_arr, PlayPlot
import numpy as np

import warnings
warnings.filterwarnings("ignore")

env = gym.make("gymnasium_arg:lunar-lander-i1", render_mode="human")
env.reset()
env.render()
# def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
#     if terminated :
#     	print("\x1b[6;30;42mSuccess!!\x1b[0m") if rew==100 else print("Crashed!!")   
#     return [rew]
play(env)