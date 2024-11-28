from isaacgym import gymapi, gymtorch
import torch
import numpy as np


class LunarLanderEnv:
    def __init__(self, num_envs=1, sim_device="cuda:0", urdf_path="../models/lander.urdf"):
        self.num_envs = num_envs
        self.sim_device = sim_device
        self.dt = 1.0 / 60.0
        self.urdf_path = urdf_path

        # Initialize Gym API
        self.gym = gymapi.acquire_gym()

        # Simulation parameters
        # sim_params = gymapi.SimParams()
        # sim_params.dt = self.dt
        # sim_params.use_gpu_pipeline = False
        # sim_params.gravity = gymapi.Vec3(0.0, -9.81, 0.0)  # Gravity applied

        sim_params = gymapi.SimParams()
        sim_params.dt = self.dt
        sim_params.use_gpu_pipeline = False
        sim_params.gravity = gymapi.Vec3(0.0, -9.81, 0.0)  # Gravity applied
        sim_params.physx.num_position_iterations = 20
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.bounce_threshold_velocity = 0.2
        sim_params.physx.max_depenetration_velocity = 1.0
        sim_params.physx.default_buffer_size_multiplier = 5


        # Create simulation
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise Exception("Failed to create simulation")

        # Create environments
        self.envs = []
        self.landers = []
        self._create_envs()

        # Viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise Exception("Failed to create viewer")

        # Set the camera position for the viewer
        cam_pos = gymapi.Vec3(0.0, 15.0, 25.0)  # Adjust the values as needed
        cam_target = gymapi.Vec3(0.0, 10.0, 0.0)  # Point towards your lander
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _create_envs(self):
        spacing = 10.0
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # Add a ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 1.0, 0.0)  # Normal pointing upwards
        plane_params.distance = 0.0  # Plane at y = 0
        plane_params.static_friction = 0.5
        plane_params.dynamic_friction = 0.5
        plane_params.restitution = 0.1
        self.gym.add_ground(self.sim, plane_params)

        # Load the URDF model
        asset_root = "."
        asset_file = self.urdf_path
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True  # Debug: Fix base link
        asset_options.armature = 0.01  # Small armature for better stability
        asset_options.flip_visual_attachments = False

        lander_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        if lander_asset is None:
            raise Exception(f"Failed to load asset from {asset_file}")

        # Create environments and place the lander in each
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, 1)
            self.envs.append(env)

            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(0.0, 10.0, 0.0)  # Start position above ground
            start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # Identity quaternion

            lander_handle = self.gym.create_actor(env, lander_asset, start_pose, "lander", i, 1)
            self.landers.append(lander_handle)



    def reset(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Acquire the root state tensor if it doesn't exist
        if not hasattr(self, "root_state_tensor"):
            self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
            self.root_states = gymtorch.wrap_tensor(self.root_state_tensor)

        # Debug: Print initial state
        print("Resetting states...")
        print("Initial state:", self.root_states)

        # Reset positions and velocities for all environments
        for i in range(self.num_envs):
            self.root_states[i, 0:3] = torch.tensor([0.0, 10.0, 0.0], device=self.sim_device)  # Position
            self.root_states[i, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.sim_device)  # Orientation
            self.root_states[i, 7:10] = 0.0  # Linear velocity
            self.root_states[i, 10:13] = 0.0  # Angular velocity

        # Commit the changes to the simulator
        self.gym.set_actor_root_state_tensor(self.sim, self.root_state_tensor)

        return self._get_observation()

    def step(self, actions):
        # Adjust velocities based on actions
        for i, action in enumerate(actions):
            if action == 0:  # Main thrust
                self.root_states[i, 7] += 0.0  # vx
                self.root_states[i, 8] += 0.5  # vy
                self.root_states[i, 9] += 0.0  # vz
            elif action == 1:  # Left thrust
                self.root_states[i, 7] += -0.1
                self.root_states[i, 8] += 0.3
                self.root_states[i, 9] += 0.0
            elif action == 2:  # Right thrust
                self.root_states[i, 7] += 0.1
                self.root_states[i, 8] += 0.3
                self.root_states[i, 9] += 0.0

        # Commit the changes to the simulator
        self.gym.set_actor_root_state_tensor(self.sim, self.root_state_tensor)

        # Step simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Get observations, rewards, and done flags
        obs = self._get_observation()
        rewards = self._get_rewards()
        dones = self._get_dones()
        return obs, rewards, dones, {}

    def render(self):
        if not self.gym.query_viewer_has_closed(self.viewer):
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
        else:
            self.close()

    def close(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def _get_observation(self):
        obs = []
        for env, lander in zip(self.envs, self.landers):
            state = self.gym.get_actor_rigid_body_states(env, lander, gymapi.STATE_ALL)

            # Extract structured data fields
            position = np.array([state['pose']['p']['x'], state['pose']['p']['y'], state['pose']['p']['z']], dtype=np.float32)
            velocity = np.array([state['vel']['linear']['x'], state['vel']['linear']['y'], state['vel']['linear']['z']], dtype=np.float32)
            angle = np.array([state['pose']['r']['x'], state['pose']['r']['y'], state['pose']['r']['z'], state['pose']['r']['w']], dtype=np.float32)
            angular_velocity = np.array([state['vel']['angular']['x'], state['vel']['angular']['y'], state['vel']['angular']['z']], dtype=np.float32)

            # Concatenate the arrays
            obs.append(np.concatenate([position, velocity, angle, angular_velocity]))

        return np.array(obs, dtype=np.float32)

    def _get_rewards(self):
        rewards = []
        for env, lander in zip(self.envs, self.landers):
            state = self.gym.get_actor_rigid_body_states(env, lander, gymapi.STATE_ALL)
            position = np.array([state['pose']['p']['x'], state['pose']['p']['y'], state['pose']['p']['z']], dtype=np.float32)
            reward = -np.linalg.norm(position)  # Negative distance from origin
            rewards.append(reward)
        return np.array(rewards)

    def _get_dones(self):
        dones = []
        for env, lander in zip(self.envs, self.landers):
            state = self.gym.get_actor_rigid_body_states(env, lander, gymapi.STATE_ALL)
            y_position = state['pose']['p']['y']
            done = y_position <= 0.0  # Y position below ground
            dones.append(done)
        return np.array(dones)


# Usage Example
env = LunarLanderEnv(num_envs=1, urdf_path="lander.urdf")
obs = env.reset()

while True: 
    actions = np.random.choice([0, 1, 2], size=1)  # Random actions
    obs, rewards, dones, info = env.step(actions)
    env.render()

env.close()
