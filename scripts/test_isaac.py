from isaacgym import gymapi
import time

# Initialize Isaac Gym
gym = gymapi.acquire_gym()

# Simulation parameters
sim_params = gymapi.SimParams()
sim_params.gravity = gymapi.Vec3(0.0, -9.81, 0.0)  # Apply gravity
sim_params.up_axis = gymapi.UP_AXIS_Y
sim_params.dt = 1 / 60.0
sim_params.substeps = 4
sim_params.physx.num_position_iterations = 20
sim_params.physx.num_velocity_iterations = 1

# Create simulation
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    raise Exception("Failed to create simulation")

# Create a ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 1.0, 0.0)  # Plane is horizontal
plane_params.distance = 0.0  # Ground at y = 0
gym.add_ground(sim, plane_params)

# Load a box asset
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False  # Allow the box to fall
box_asset = gym.create_box(sim, 1.0, 1.0, 1.0, asset_options)

# Create an environment
env = gym.create_env(sim, gymapi.Vec3(-5.0, 0.0, -5.0), gymapi.Vec3(5.0, 5.0, 5.0), 1)

# Add the box actor
start_pose = gymapi.Transform()
start_pose.p = gymapi.Vec3(0.0, 5.0, 0.0)  # Start above the ground
start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # No rotation
box_actor = gym.create_actor(env, box_asset, start_pose, "box", 0, 1)

# Create a viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# Camera setup
cam_pos = gymapi.Vec3(5.0, 5.0, 10.0)
cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# Simulate
print("Starting simulation...")

while not gym.query_viewer_has_closed(viewer):
    # Step the physics simulation
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Step the graphics simulation
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Pause to sync to the real-time simulation
    time.sleep(1 / 60.0)

# Cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
