from isaacgym import gymapi

# Initialize gym
gym = gymapi.acquire_gym()

# Set simulation parameters
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0  # Simulation timestep
sim_params.substeps = 2

# Configure the simulation device
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.use_gpu_pipeline = True  # Use GPU for physics simulation

# Create the simulation
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Create a viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# Main simulation loop
while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

# Clean up
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

# import isaacgym
# import isaacgymenvs
# import torch

# envs = isaacgymenvs.make(
#     seed=0, 
#     task="Ant", 
#     num_envs=2000, 
#     sim_device="cuda:0", 
#     rl_device="cuda:0", 
#     # headless=True  # Add this to suppress viewer
# )

# print("Observation space is", envs.observation_space)
# print("Action space is", envs.action_space)
# obs = envs.reset()
# for _ in range(20):
# 	obs, reward, done, info = envs.step(
# 		torch.rand((2000,)+envs.action_space.shape, device="cuda:0")
# 	)
