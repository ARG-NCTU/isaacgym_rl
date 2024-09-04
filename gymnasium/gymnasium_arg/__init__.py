from gymnasium.envs.registration import register

from gymnasium_arg.envs.lunarlander_i1 import LunarLanderI1


register(
    id="lunar-lander-i1",
    entry_point="gymnasium_arg.envs:LunarLanderI1",
    max_episode_steps=4096,
)

