# asset parameters for the simulator above


class control_allocator_config:
    num_motors = 8
    force_application_level = "motor_link"
    # "motor_link" or "root_link" to apply forces at the root link or at the individual motor links

    motor_mask = [1 + 8 + i for i in range(0, 8)]
    motor_directions = [1, -1, 1, -1, 1, -1, 1, -1]

    allocation_matrix = [[ 5.55111512e-17, -3.21393805e-01, -4.54519478e-01, -3.42020143e-01,
                        9.69846310e-01,  3.42020143e-01,  8.66025404e-01, -7.54406507e-01],
                        [ 1.00000000e+00, -3.42020143e-01, -7.07106781e-01,  0.00000000e+00,
                        -1.73648178e-01,  9.39692621e-01,  5.00000000e-01, -1.73648178e-01],
                        [ 1.66533454e-16, -8.83022222e-01,  5.41675220e-01,  9.39692621e-01,
                        1.71010072e-01,  1.11022302e-16,  1.11022302e-16,  6.33022222e-01],
                        [ 1.75000000e-01,  1.23788742e-01, -5.69783368e-02,  1.34977168e-01,
                        3.36959042e-02, -2.66534135e-01, -7.88397460e-02, -2.06893989e-02],
                        [ 1.00000000e-02,  2.78845133e-01, -4.32852308e-02, -2.72061766e-01,
                        -1.97793856e-01,  8.63687139e-02,  1.56554446e-01, -1.71261290e-01],
                        [ 2.82487373e-01, -1.41735490e-01, -8.58541103e-02,  3.84858939e-02,
                        -3.33468026e-01,  8.36741468e-02,  8.46777988e-03, -8.74336259e-02]]

    # here, the allocation matrix is computed (by the user) to from the URDF files of the robot
    # to map the effect of motor forces on the net force and torque acting on the robot.

class motor_model_config:
    motor_time_constant_min = 0.01
    motor_time_constant_max = 0.03
    max_thrust = 5.0
    min_thrust = -5.0
    max_thrust_rate = 100.0
    thrust_to_torque_ratio = 0.01 # thrust to torque ratio is related to inertia matrix dont change

# other parameters for the robot below