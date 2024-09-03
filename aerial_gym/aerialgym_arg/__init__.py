from aerial_gym.registry.sim_registry import sim_config_registry
from aerial_gym.registry.env_registry import env_config_registry

from aerialgym_arg.envs.lunarlander.lunarlander_sim import LunarPhyx
from aerialgym_arg.envs.lunarlander.lunarlander_env import LunarEnvCfg

def init_package():
    print("Initializing package aerialgym_arg")
    sim_config_registry.register(sim_name="lunar_phyx", sim_config=LunarPhyx)
    env_config_registry.register(env_name="lunar_env", env_config=LunarEnvCfg)

init_package()