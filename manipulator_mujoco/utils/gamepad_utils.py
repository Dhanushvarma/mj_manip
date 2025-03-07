import numpy as np
import time

from manipulator_mujoco.gamepad.gamepad import available
from manipulator_mujoco.gamepad.controllers import PS4


def get_gamepad_action(gamepad):
    """
    returns the action, control_active and terminate
    action : array of size 7, where first 6 elements are the position input and the last element is the gripper input
    control_active : boolean, True if the R2 key is pressed, False otherwise
    terminate : boolean, True if the CIRCLE key is pressed, False otherwise
    """

    action = np.zeros((7,))

    # position deltas
    position_gain = 0.01
    action[0] = position_gain * gamepad.axis("LEFT-Y")  # x
    action[1] = position_gain * gamepad.axis("LEFT-X")  # y
    action[2] = -position_gain * gamepad.axis("RIGHT-Y")  # z

    # orientation deltas set to zero
    action[3] = 0
    action[4] = 0
    action[5] = 0

    # gripper control
    action[6] = 1 if gamepad.axis("L2") == 1 else 0

    # terminate
    terminate = True if gamepad.beenPressed("CIRCLE") else False

    # optional : control active
    control_active = True if gamepad.axis("R2") == 1 else False

    return action, control_active, terminate


def connect_gamepad():
    if not available():
        print("Please connect your gamepad...")
        while not available():
            time.sleep(1.0)
    gamepad = PS4()
    print("Gamepad connected")
    gamepad.startBackgroundUpdates()
    return gamepad
