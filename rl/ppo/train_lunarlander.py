import sys, os, time
import threading
import gymnasium as gym
import warnings
from gymnasium_arg.envs.lunarlander_i1 import task_config
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnRewardThreshold
from datetime import date
import add_path
from isaac_vec_env import IsaacVecEnv
from isaac_extractor import CustomFeatureExtractor
import torch


def check_input():
    input("Press Enter to stop rendering...\n")
    global stop_rendering
    stop_rendering = True

# Start a separate thread to listen for input
stop_rendering = False
input_thread = threading.Thread(target=check_input)
input_thread.start()



warnings.filterwarnings("ignore")
policy_kwargs = dict(features_extractor_class=CustomFeatureExtractor)

today = date.today()
checkpoint_callback = CheckpointCallback(
  save_freq=100000,
  save_path="./logs/",
  name_prefix="dp_"+str(today),
  save_replay_buffer=True,
  save_vecnormalize=True,
)
num_envs = 10
task = task_config()
task.headless = False
env = gym.make("gymnasium_arg:lunar-lander-i1", task_config=task, render_mode="rgb_array", num_envs=num_envs)
env.reset()
env.render()
# Rendering loop
while True:
    obs, reward, terminated, truncated, info = env.step(torch.zeros(num_envs, 4, device='cuda'))
    env.render()
    for i in range(len(terminated)):
        if terminated[i] or truncated[i]:
            env.reset_idx(i)
    # Check if the user pressed Enter to stop rendering
    if stop_rendering:
        print("Stopping rendering...")
        break
    

env.reset()
vec_env = IsaacVecEnv(env)
# Parallel environments
model = PPO("MultiInputPolicy", vec_env,
            verbose=1, 
            policy_kwargs=policy_kwargs, 
            learning_rate=1e-6,
            batch_size=128,
            n_steps=4096,
            n_epochs=10,
            ent_coef=0.01,
            device='cuda',
            tensorboard_log='tb_ppo')

model.learn(total_timesteps=20_000_000, tb_log_name='tb_ppo', callback=checkpoint_callback)
model.save("ppo_lunarlander")
