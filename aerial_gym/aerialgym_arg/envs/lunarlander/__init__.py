from aerial_gym.registry.sim_registry import sim_config_registry

from lunarlander_sim import LunarPhyx

sim_config_registry.register(sim_name="lunar_phyx", sim_config=LunarPhyx)
