from aerial_gym.registry.sim_registry import sim_config_registry
from aerial_gym.registry.env_registry import env_config_registry

from .envs.lunarlander.lunarlander_sim import LunarLanderPhyx
from .envs.lunarlander.lunarlander_env import LunarLanderEnvCfg
from .envs.lunarlander.lunarlander_sensor import LunarLanderSensorCfg
from .envs.lunarlander.lunarlander_task import LunarLanderTask

def registring_lunarlander_package():
    print("Registring lunarlander config to aerialgym_arg")
    sim_config_registry.register(sim_name="lunarlander_phyx", sim_config=LunarLanderPhyx)
    env_config_registry.register(env_name="lunarlander_env", env_config=LunarLanderEnvCfg)

registring_lunarlander_package()