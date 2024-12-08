from isaacgym import gymapi, gymtorch
from pykeyboard import KeyboardInput
import torch

class CarSimulation:
    def __init__(self, num_envs, motor_names, dt, device="cuda:0"):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device

        # Initialize Isaac Gym
        self.gym = gymapi.acquire_gym() # acquire_gym() is a function that returns a gym object
        self.sim = None
        self.viewer = None
        self.envs = []
        self.actors = []
        self.motor_names = motor_names  # List of joint names for the prismatic joints
        self.motor_dof_indices = []  # List of lists: DOF indices per actor

        # Simulation parameters
        self.sim_params = gymapi.SimParams()
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)  # Gravity along -Z
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.dt = dt

        self.create_sim()
        self.create_viewer()

    def create_sim(self):
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, self.sim_params) # create_sim() is a function that creates a simulation object

        # Add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)  # Z-up
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.5
        self.gym.add_ground(self.sim, plane_params)

        # Add robot assets
        spacing = 2.0
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "."
        asset_file = "car_gym.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False  # Allow robot to move
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, self.num_envs)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 3.0)
            pose.r = gymapi.Quat(0.707, 0.0, 0.0, 0.707)
            actor = self.gym.create_actor(env, robot_asset, pose, "robot", i, 1)
            self.envs.append(env)
            self.actors.append(actor)

            # Configure DOF properties for effort control
            dof_props = self.gym.get_actor_dof_properties(env, actor)
            for j in range(len(dof_props)):
                dof_props['driveMode'][j] = gymapi.DOF_MODE_EFFORT
                dof_props['stiffness'][j] = 0.0
                dof_props['damping'][j] = 0.0
            self.gym.set_actor_dof_properties(env, actor, dof_props)

            # Get DOF indices for the specified motor names
            dof_dict = self.gym.get_actor_dof_dict(env, actor)
            motor_indices = []
            for motor_name in self.motor_names:
                if motor_name in dof_dict:
                    motor_indices.append(dof_dict[motor_name])
                else:
                    raise ValueError(f"Motor name '{motor_name}' not found in actor's DOFs.")
            self.motor_dof_indices.append(motor_indices)

    def create_viewer(self):
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise ValueError("Failed to create viewer")

    def step(self, actions):
        """
        Apply actions to the motors and simulate one step.
        """
        for i, env in enumerate(self.envs):
            actor = self.actors[i]

            # Initialize DOF efforts to zeros for all DOFs
            num_dofs = self.gym.get_actor_dof_count(env, actor)
            dof_efforts = torch.zeros(num_dofs, device=self.device)

            # Apply actions to the specified motor DOFs
            for j, dof_index in enumerate(self.motor_dof_indices[i]):
                if dof_index < num_dofs: # Ensure index is within bounds
                    dof_efforts[dof_index] = actions[i, j]
                else:
                    print(f"Warning: DOF index {dof_index} out of bounds for actor {i}.")

            # Convert the PyTorch tensor to a NumPy array
            dof_efforts_np = dof_efforts.cpu().numpy()

            # Apply the efforts to the actor
            self.gym.apply_actor_dof_efforts(env, actor, dof_efforts_np)

        # Simulate one step
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Update the viewer
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

    def cleanup(self):
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)


# Example configuration
motor_names = [
    "wheel1_RevoluteJoint",
    "wheel2_RevoluteJoint",
    "wheel3_RevoluteJoint",
    "wheel4_RevoluteJoint",
]
dt = 0.02

# Initialize Keyboard Input
keyboard_input = KeyboardInput()
num_envs = 1

if __name__ == "__main__":
    motor_sim = CarSimulation(num_envs, motor_names, dt)
    step = 0
    try:
        while True:
            step += 1
            keyboard_input.handle_events()
            input_key = keyboard_input.get_actions()
            actions = torch.zeros(1, 4, device="cuda:0")

            ################ Write your code here ################
            '''
            Task: Map the input [up, down, left, right] to the Car Movements [forward, backward, left, right]
            Input Variable: input_key: 1x4 tensor [up, down, left, right]
            Output Variable: actions: 1x4 tensor [Wheel1, Wheel2, Wheel3, Wheel4]
            '''


            ######################################################
            
            actions = actions.repeat(num_envs, 1)
            motor_sim.step(actions)
            print(f"Step {step + 1}: Actions: {actions.cpu().numpy()}")

            # Exit if viewer is closed or pygame window is closed
            if motor_sim.gym.query_viewer_has_closed(motor_sim.viewer) or not keyboard_input.running:
                break
    finally:
        motor_sim.cleanup()
        keyboard_input.quit()
