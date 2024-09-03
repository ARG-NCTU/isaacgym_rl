import numpy as np
from aerial_gym.registry.sim_registry import sim_registry
from aerial_gym.simulation.sim_params import BaseSimParams, BaseSimConfig

class LunarLanderParamsFallingForwards(BaseSimParams):
    class sim(BaseSimConfig.sim):
        dt = 0.01  # Custom Parameter
        gravity = [0.0, 0.0, -1.0]


# register your custom class here
sim_registry.register_sim_params("lunar_world", LunarLanderParamsFallingForwards)

### use the registered class further in the code to spawn a simulation ###


class LunarLanderSimulation:
    def __init__(self, config):
        self.gravity = config['physics']['gravity']
        self.fuel = config['fuel']
        self.position = np.array([0.0, 10.0])
        self.velocity = np.array([0.0, 0.0])
        self.thrust_main = config['components']['main_thruster']['parameters']['max_thrust']
        self.thrust_side = config['components']['side_thrusters']['parameters']['max_thrust']

    def apply_thrust(self, main_thrust, side_thrust):
        if self.fuel <= 0:
            return

        # Apply main thrust
        self.velocity[1] += main_thrust * self.thrust_main
        self.velocity[0] += side_thrust * self.thrust_side

        # Simulate gravity
        self.velocity[1] -= self.gravity

        # Update position
        self.position += self.velocity

        # Fuel consumption
        self.fuel -= abs(main_thrust) + abs(side_thrust)

    def check_landing(self, landing_pad):
        x_landing = landing_pad['position'][0] - landing_pad['size'][0]/2 <= self.position[0] <= landing_pad['position'][0] + landing_pad['size'][0]/2
        y_landing = self.position[1] <= landing_pad['position'][1] + landing_pad['size'][1]/2
        return x_landing and y_landing and np.allclose(self.velocity, [0, 0], atol=0.5)

    def get_state(self):
        return {
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'fuel': self.fuel
        }
