import gymnasium as gym
from gym.spaces import Box, Discrete
from isaacgym import gymapi, gymtorch
import torch
import numpy as np
import math

# Constants
FPS = 50
SCALE = 30.0

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0

# Lander geometry scaled appropriately
LANDER_SIZE = 0.2  # Half-length in each dimension

# Reward parameters
REWARD_CRASH = -100.0
REWARD_LAND = +100.0
REWARD_MAIN_ENGINE = -0.3
REWARD_SIDE_ENGINE = -0.03

class SimplifiedLunarLanderEnv:
    def __init__(self, num_envs=1, sim_device="cpu", gravity=-10.0, continuous=False):
        self.num_envs = num_envs
        self.sim_device = sim_device
        self.gravity = gravity
        self.continuous = continuous
        self.dt = 1.0 / FPS

        # Initialize Gym API
        self.gym = gymapi.acquire_gym()

        # Simulation parameters
        sim_params = gymapi.SimParams()
        sim_params.dt = self.dt
        sim_params.gravity = gymapi.Vec3(0.0, self.gravity, 0.0)
        sim_params.use_gpu_pipeline = False

        # Create simulation
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise Exception("Failed to create simulation")

        # Create environments
        self.envs = []
        self.landers = []
        self._create_envs()

        # Define lander mass
        self.lander_mass = 1.0  # Adjust as needed

        # Define observation and action spaces
        low = np.array([-1.5, -1.5, -5.0, -5.0, -math.pi, -5.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.5, 1.5, 5.0, 5.0, math.pi, 5.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = Box(low, high, dtype=np.float32)

        if self.continuous:
            self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        else:
            self.action_space = Discrete(4)

        # Viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise Exception("Failed to create viewer")

        # Set the camera position for the viewer
        cam_pos = gymapi.Vec3(0.0, 15.0, 25.0)  # Adjust as needed
        cam_target = gymapi.Vec3(0.0, 10.0, 0.0)  # Point towards the lander
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # Initialize previous shaping for reward calculation
        self.prev_shaping = np.zeros(self.num_envs, dtype=np.float32)

    def _create_envs(self):
        spacing = 10.0
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, 1)
            self.envs.append(env)

            # Create the lander as a single box
            lander_asset_options = gymapi.AssetOptions()
            lander_asset_options.fix_base_link = False

            lander_asset = self.gym.create_box(
                self.sim,
                LANDER_SIZE,  # Half-length in x
                LANDER_SIZE,  # Half-length in y
                LANDER_SIZE,  # Half-length in z
                lander_asset_options
            )

            if lander_asset is None:
                raise Exception("Failed to create lander asset")

            # Starting pose
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(0.0, 10.0, 0.0)
            start_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, 0.0)

            # Create lander actor
            lander_handle = self.gym.create_actor(env, lander_asset, start_pose, f"lander_{i}", i, 1)
            self.landers.append(lander_handle)

    def reset(self):
        # Reset simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Acquire the root state tensor if it doesn't exist
        if not hasattr(self, "root_state_tensor"):
            self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
            self.root_states = gymtorch.wrap_tensor(self.root_state_tensor).to(self.sim_device)

        # Reset positions and velocities for all environments
        for i in range(self.num_envs):
            # Reset lander
            self.root_states[i, 0:3] = torch.tensor([0.0, 10.0, 0.0], device=self.sim_device)  # Position
            self.root_states[i, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.sim_device)  # Orientation (quaternion)
            self.root_states[i, 7:10] = torch.zeros(3, device=self.sim_device)  # Linear velocity
            self.root_states[i, 10:13] = torch.zeros(3, device=self.sim_device)  # Angular velocity

        # Commit the changes to the simulator
        self.gym.set_actor_root_state_tensor(self.sim, self.root_state_tensor)

        # Reset previous shaping
        self.prev_shaping = np.zeros(self.num_envs, dtype=np.float32)

        return self._get_observation()

    def step(self, actions):
        self.current_actions = actions  # Store current actions for reward calculation

        # Convert actions to torch tensor
        if self.continuous:
            actions_tensor = torch.tensor(actions, dtype=torch.float32, device=self.sim_device)
        else:
            actions_tensor = torch.tensor(actions, dtype=torch.int32, device=self.sim_device)

        # Ensure root state tensor is acquired
        if not hasattr(self, "root_state_tensor"):
            self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
            self.root_states = gymtorch.wrap_tensor(self.root_state_tensor).to(self.sim_device)

        # Modify velocities based on actions
        for i in range(self.num_envs):
            idx = i  # Actor index

            # Get current velocities
            linear_vel = self.root_states[idx, 7:10]

            # Initialize force vector
            force = torch.zeros(3, device=self.sim_device)

            if self.continuous:
                main_throttle = torch.clamp(actions_tensor[i, 0], -1.0, 1.0).item()
                lateral_throttle = torch.clamp(actions_tensor[i, 1], -1.0, 1.0).item()

                # Main engine force
                if main_throttle > 0.0:
                    force[1] += MAIN_ENGINE_POWER * ((main_throttle + 1.0) * 0.5)

                # Side engines
                if lateral_throttle < -0.5:
                    force[0] -= SIDE_ENGINE_POWER * ((-lateral_throttle + 1.0) * 0.5)
                elif lateral_throttle > 0.5:
                    force[0] += SIDE_ENGINE_POWER * ((lateral_throttle + 1.0) * 0.5)

            else:
                action = int(actions_tensor[i].item())
                if action == 1:
                    # Fire left thruster
                    force[0] -= SIDE_ENGINE_POWER
                    force[1] += MAIN_ENGINE_POWER
                elif action == 2:
                    # Fire main thruster
                    force[1] += MAIN_ENGINE_POWER
                elif action == 3:
                    # Fire right thruster
                    force[0] += SIDE_ENGINE_POWER
                    force[1] += MAIN_ENGINE_POWER
                # Action 0: Do nothing

            # Compute accelerations
            mass = self.lander_mass  # Already defined
            linear_acc = force / mass

            # Update velocities
            delta_linear_vel = linear_acc * self.dt
            self.root_states[idx, 7:10] = linear_vel + delta_linear_vel

            # Optionally handle angular velocities if torque is computed

        # Set the modified root state tensor back to the simulator
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

    def _quat_to_euler(self, quat):
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        w, x, y, z = quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return [roll, pitch, yaw]

    def _get_observation(self):
        obs = []
        for i in range(self.num_envs):
            # Get lander state
            main_state = self.root_states[i].cpu().numpy()
            pos = main_state[0:3]
            vel = main_state[7:10]
            quat = main_state[3:7]
            angle = self._quat_to_euler(quat)
            angular_vel = main_state[10:13]

            # Normalize position and velocity
            pos_x = pos[0] / 1.5
            pos_y = (pos[1] - 10.0) / 1.5  # Adjust since legs are removed
            vel_x = vel[0] * (1.5 / FPS)
            vel_y = vel[1] * (1.5 / FPS)

            # Normalize angle and angular velocity
            angle = angle[2]  # Yaw angle
            angular_vel = angular_vel[2] * (20.0 / FPS)

            # No leg contacts in simplified environment
            leg1_contact = 0.0
            leg2_contact = 0.0

            state = [pos_x, pos_y, vel_x, vel_y, angle, angular_vel, leg1_contact, leg2_contact]
            obs.append(state)

        return np.array(obs, dtype=np.float32)

    def _get_rewards(self):
        rewards = []
        for i in range(self.num_envs):
            state = self._get_observation()[i]
            pos = state[0:2]
            vel = state[2:4]
            angle = state[4]
            angular_vel = state[5]
            leg1 = state[6]
            leg2 = state[7]

            # Shaping rewards
            shaping = (
                -100.0 * np.sqrt(pos[0] ** 2 + pos[1] ** 2)
                - 100.0 * np.sqrt(vel[0] ** 2 + vel[1] ** 2)
                - 100.0 * abs(angle)
                + 10.0 * leg1
                + 10.0 * leg2
            )

            if self.prev_shaping[i] is not None:
                reward = shaping - self.prev_shaping[i]
            else:
                reward = shaping
            self.prev_shaping[i] = shaping

            # Penalize fuel usage based on actions
            if self.continuous:
                main_throttle = torch.clamp(torch.tensor(self.current_actions[i, 0]), -1.0, 1.0).item()
                lateral_throttle = torch.clamp(torch.tensor(self.current_actions[i, 1]), -1.0, 1.0).item()

                # Main engine usage
                if main_throttle > 0.0:
                    reward += REWARD_MAIN_ENGINE * ((main_throttle + 1.0) * 0.5)  # Scaled 0.5 to 1.0

                # Left thruster usage
                if lateral_throttle < -0.5:
                    reward += REWARD_SIDE_ENGINE * ((-lateral_throttle + 1.0) * 0.5)  # Scaled 0.5 to 1.0

                # Right thruster usage
                if lateral_throttle > 0.5:
                    reward += REWARD_SIDE_ENGINE * ((lateral_throttle + 1.0) * 0.5)  # Scaled 0.5 to 1.0
            else:
                action = int(self.current_actions[i])
                if action in [1, 3]:
                    reward += REWARD_SIDE_ENGINE
                if action == 2:
                    reward += REWARD_MAIN_ENGINE

            # Termination rewards
            done = self._get_dones()[i]
            if done:
                if state[1] <= 0.0:
                    reward = REWARD_CRASH
                else:
                    reward = REWARD_LAND

            rewards.append(reward)

        return np.array(rewards, dtype=np.float32)

    def _get_dones(self):
        dones = []
        for i in range(self.num_envs):
            state = self._get_observation()[i]
            pos = state[0:2]
            done = False

            # Crash conditions
            if pos[1] <= 0.0 or abs(pos[0]) >= 1.0:
                done = True

            # Retrieve actor's linear and angular velocities from the root state tensor
            # Assuming body_index=0 since SimplifiedLunarLanderEnv has a single body
            actor_state = self.root_states[i].cpu().numpy()
            linear_vel = actor_state[7:10]    # Indices 7, 8, 9 correspond to linear velocity
            angular_vel = actor_state[10:13]  # Indices 10, 11, 12 correspond to angular velocity

            # Define velocity thresholds
            linear_vel_threshold = 0.1  # Adjust based on desired sensitivity
            angular_vel_threshold = 0.1 # Adjust based on desired sensitivity

            # Calculate the magnitude of velocities
            linear_speed = np.linalg.norm(linear_vel)
            angular_speed = np.linalg.norm(angular_vel)

            # If both linear and angular speeds are below thresholds, consider the actor as "asleep"
            if linear_speed < linear_vel_threshold and angular_speed < angular_vel_threshold:
                done = True

            dones.append(done)
        return np.array(dones, dtype=np.bool_)

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

# Main script
if __name__ == "__main__":
    import time

    # Initialize environment
    env = SimplifiedLunarLanderEnv(num_envs=10, sim_device="cpu", continuous=False)  # Set continuous=True for continuous actions
    obs = env.reset()

    for step in range(10000):
        # Example: Random actions
        if env.continuous:
            actions = np.random.uniform(-1.0, 1.0, size=(env.num_envs, 2)).astype(np.float32)
        else:
            actions = np.random.choice([0, 1, 2, 3], size=env.num_envs)

        obs, rewards, dones, info = env.step(actions)
        env.render()

        if dones.any():
            print(f"Episode terminated with rewards: {rewards}")
            obs = env.reset()

        # Optional: Sleep to match real-time (not necessary for training)
        time.sleep(env.dt)

    env.close()