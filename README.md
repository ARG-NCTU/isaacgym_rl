# Lab: IsaacGym with Keyboard Control via PyGame

## Clone this repo
```bash
cd ~
git clone https://github.com/ARG-NCTU/isaacgym_rl.git
```
---

## Lab Overview

In this lab, students will learn how to integrate keyboard inputs with robotic car simulation in Isaac Gym. The objective is to map user commands from the keyboard (`up`, `down`, `left`, `right`) to motor control actions for a simulated car. This lab combines robotics simulation, keyboard input handling, and control logic.

---

## Learning Objectives

1. **Understand Simulation Basics:**
   - Set up a robotic car simulation using NVIDIA Isaac Gym.
   - Work with DOF (Degrees of Freedom) properties and effort control.

2. **Integrate User Inputs:**
   - Use keyboard inputs to control the car's motors dynamically.

3. **Develop Control Logic:**
   - Map keyboard inputs (`up`, `down`, `left`, `right`) to the car's movement (`forward`, `backward`, `turn left`, `turn right`).

---

## Lab Task

### Task: Map Keyboard Inputs to Motor Actions

- In the main loop of the script, map the input tensor `[up, down, left, right]` to the motor actions `[Wheel1, Wheel2, Wheel3, Wheel4]` such that:
  - **Up (`up`)**: All wheels move forward.
  - **Down (`down`)**: All wheels move backward.
  - **Left (`left`)**: Left wheels move backward, and right wheels move forward (turn left).
  - **Right (`right`)**: Right wheels move backward, and left wheels move forward (turn right).

---

## Implementation Details

### Simulation Initialization

The `CarSimulation` class handles:
- Creating a physics-based simulation with Isaac Gym.
- Loading a car asset from a URDF file (`car_gym.urdf`).
- Setting DOF properties for motor control.

### Keyboard Input Integration

The `KeyboardInput` class from the `pykeyboard` library provides:
- A simple interface to capture keyboard inputs (`up`, `down`, `left`, `right`).
- A `get_actions()` method that returns the action tensor `[up, down, left, right]`.

### Main Task: Map Keyboard Inputs to Motor Actions

Modify the section marked with:

```python
################ Write your code here ################
'''
Task: Map the input [up, down, left, right] to the Car Movements [forward, backward, left, right]
Input Variable: actions: 1x4 tensor [up, down, left, right]
Output Variable: actions: 1x4 tensor [Wheel1, Wheel2, Wheel3, Wheel4]
'''
######################################################
```

### Test Your Code
- Modify the code at **~/isaacgym_rl/scripts/lab_car.py**.
- Enter Docker then test it.
```bash
cd ~/isaacgym_rl
source Docker/run.sh
cd scripts
python3 lab_car.py
```
