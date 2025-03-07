import gymnasium
import numpy as np
import time

from manipulator_mujoco.gamepad.controllers import PS4
from manipulator_mujoco.utils.gamepad_utils import get_gamepad_action, connect_gamepad

# Create the environment with rendering in human mode
env = gymnasium.make(
    "manipulator_mujoco/WidowEnv-v0", render_mode="human", render_backend="mjviewer"
)

# Reset the environment with a specific seed for reproducibility
observation, info = env.reset(seed=42)

# create a gamepad instance
gamepad = connect_gamepad()

# Run simulation for a fixed number of steps
while True:

    # get action from gamepad
    action, control_active, _ = get_gamepad_action(gamepad)
    grip = action[-1]
    action = action[:-1]
    action = (action, grip)

    # Take a step in the environment using the chosen action
    observation, reward, terminated, truncated, info = env.step(action)

    # Check if the episode is over (terminated) or max steps reached (truncated)
    if terminated or truncated:
        time.sleep(2)

        # If the episode ends or is truncated, reset the environment
        observation, info = env.reset()

# Close the environment when the simulation is done
env.close()
