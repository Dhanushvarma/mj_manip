import time
import numpy as np
import os
import cv2  # Added for OpenCV window rendering
from loguru import logger
from dm_control import mjcf
import gymnasium as gym
from gymnasium import spaces
from manipulator_mujoco.arenas import StandardArena
from manipulator_mujoco.robots import Arm
from manipulator_mujoco.props import Primitive
from manipulator_mujoco.mocaps import Target
from manipulator_mujoco.controllers import OperationalSpaceController


class WidowEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,  # Set a proper render FPS
    }

    def __init__(self, render_mode=None):
        # Observation space with reasonable image dimensions
        self.observation_space = spaces.Dict(
            {
                "pinch_site_pose": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
                ),  # pos(3) + quat(4)
                "joint_pose": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64
                ),  # 8 joint positions
                "frontview_image": spaces.Box(
                    low=0, high=255, shape=(1080, 1920, 3), dtype=np.uint8
                ),  # RGB image with useful dimensions
                "topview_image": spaces.Box(
                    low=0, high=255, shape=(1080, 1920, 3), dtype=np.uint8
                ),  # RGB image with useful dimensions
            }
        )

        # Action space for operational space control (dx, dy, dz, droll, dpitch, dyaw, grip)
        continuous_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float64)
        discrete_space = spaces.Discrete(2)
        self.action_space = spaces.Tuple((continuous_space, discrete_space))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._render_mode = render_mode

        ############################
        # create MJCF model
        ############################

        # checkerboard floor
        self._arena = StandardArena()

        # mocap target that OSC will try to follow
        self._target = Target(self._arena.mjcf_model)

        # Widow Arm
        self._arm = Arm(
            xml_path=os.path.join(
                os.path.dirname(__file__),
                "../assets/robots/trossen_wx250s/wx250s.xml",
            ),
            eef_site_name="pinch_site",
            attachment_site_name="attachment_site",
        )

        # red box
        self._red_box = Primitive(
            type="box",
            size=[0.02, 0.02, 0.02],
            pos=[0, 0, 0.02],
            rgba=[1, 0, 0, 1],
            friction=[1, 0.3, 0.0001],
        )
        self._green_box = Primitive(
            type="box",
            size=[0.02, 0.02, 0.02],
            pos=[0, 0, 0.02],
            rgba=[0, 1, 0, 1],
            friction=[1, 0.3, 0.0001],
        )

        # attach arm to arena
        self._arena.attach(self._arm.mjcf_model, pos=[0, 0, 0])

        # attach boxes to arena as free joint
        self._arena.attach_free(self._red_box.mjcf_model, pos=[0.15, 0.15, 0])
        self._arena.attach_free(self._green_box.mjcf_model, pos=[0.15, -0.15, 0])

        # attach cameras to arena
        self._arena.attach_camera(
            name="frontview", pos=[1.15, 0, 0.2125], quat=[0.5, 0.5, 0.5, 0.5]
        )
        self._arena.attach_camera(
            name="topview", pos=[0.38, 0, 0.8], quat=[0.7071068, 0, 0, 0.7071068]
        )

        # generate model
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        # increase camera buffer size
        self._physics.model.vis.global_.offwidth = 1920
        self._physics.model.vis.global_.offheight = 1080

        # set up OSC controller with appropriate parameters for Widow Arm
        self._controller = OperationalSpaceController(
            physics=self._physics,
            joints=self._arm.joints,
            eef_site=self._arm.eef_site,
            min_effort=-50.0,
            max_effort=50.0,
            kp=100,
            ko=100,
            kv=10,
            vmax_xyz=0.5,
            vmax_abg=1.0,
        )

        # for time keeping
        self._timestep = self._physics.model.opt.timestep
        self._step_start = None

        # Initialize OpenCV window if in human render mode
        self._cv_window_initialized = False
        if self._render_mode == "human":
            cv2.namedWindow("Widow Arm Simulation - Front View", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("Widow Arm Simulation - Front View", 1080, 480)
            cv2.setWindowProperty(
                "Widow Arm Simulation - Front View",
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
            self._cv_window_initialized = True

        # Track distances and goal achievement
        self._target_location = np.zeros(3)
        self._current_distance = None
        self._prev_distance = None
        self._goal_achieved = False

        # Store the current frame for rendering
        self._current_frame = None

    def _get_obs(self) -> dict:
        # Get pinch site pose (position and orientation)
        pinch_site_pos = self._physics.bind(self._arm.eef_site).xpos.copy()
        pinch_site_quat = self._physics.bind(self._arm.eef_site).quat.copy()
        pinch_site_pose = np.concatenate([pinch_site_pos, pinch_site_quat])

        # Get joint positions
        joint_pose = self._physics.bind(self._arm.joints).qpos.copy()

        # Get camera images with meaningful dimensions
        frontview_image = self._physics.render(
            height=1080, width=1920, camera_id="frontview"
        )
        topview_image = self._physics.render(
            height=1080, width=1920, camera_id="topview"
        )

        # Store the frontview image for rendering
        self._current_frame = frontview_image

        return {
            "pinch_site_pose": pinch_site_pose,
            "joint_pose": joint_pose,
            "frontview_image": frontview_image,
            "topview_image": topview_image,
        }

    def _get_info(self) -> dict:
        # Provide useful information for debugging and monitoring
        if False:
            info = {
                "distance_to_target": self._current_distance,
                "goal_achieved": self._goal_achieved,
                "red_box_pos": self._physics.bind(
                    self._red_box.mjcf_model.find_all("geom")[0]
                ).pos.copy(),
                "green_box_pos": self._physics.bind(
                    self._green_box.mjcf_model.find_all("geom")[0]
                ).pos.copy(),
                "ee_position": self._physics.bind(self._arm.eef_site).xpos.copy(),
            }
        return {}

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)

        print("Reset is occuring !")

        # Use the seed for randomization
        self.np_random = np.random.RandomState(seed)

        # reset physics
        with self._physics.reset_context():
            # put arm in a reasonable starting position
            self._physics.bind(self._arm.joints).qpos = [
                0,
                -0.96,
                1.16,
                0,
                -0.3,
                0,
                0.015,
                -0.015,
            ]

            # randomize box positions
            red_box_pos = [
                self.np_random.uniform(0.10, 0.20),
                self.np_random.uniform(0.10, 0.20),
                0.02,
            ]
            green_box_pos = [
                self.np_random.uniform(0.10, 0.20),
                self.np_random.uniform(-0.10, -0.20),
                0.02,
            ]
            self._physics.bind(self._red_box.geom).xpos = red_box_pos
            self._physics.bind(self._green_box.geom).xpos = green_box_pos

            # put target in a reasonable starting position
            target_pos = [0.5, 0, 0.1]  # Raised z position for better visibility

            self._target.set_mocap_pose(
                self._physics, position=target_pos, quaternion=[1, 0, 0, 0]
            )

            # Store target location for reward calculation
            self._target_location = target_pos

        # Get initial observation
        observation = self._get_obs()

        # Calculate initial distance to target
        ee_pos = observation["pinch_site_pose"][:3]
        self._current_distance = np.linalg.norm(ee_pos - self._target_location)
        self._prev_distance = self._current_distance

        # Reset goal achievement status
        self._goal_achieved = False

        if False:
            # Render the initial state if in human mode
            if self._render_mode == "human":
                self._render_frame()

        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> tuple:

        # hack: to access sep
        grip = action[-1]
        action = action[0]

        # Use the action to update the target pose
        current_ee_pos = self._physics.bind(self._arm.eef_site).xpos.copy()
        current_ee_quat = self._physics.bind(self._arm.eef_site).quat.copy()

        # Update position target based on action (dx, dy, dz)
        new_target_pos = current_ee_pos + action[:3]

        # TODO: use angle input to update target

        # TODO: clamp to workspace limits

        # Update the mocap target pose
        self._target.set_mocap_pose(
            self._physics, position=new_target_pos, quaternion=current_ee_quat
        )

        # Get the updated target pose
        target_pose = self._target.get_mocap_pose(self._physics)

        # Run OSC controller to move to target pose
        self._controller.run(target_pose, grip=grip)

        # Step physics
        self._physics.step()

        # Calculate distance to target location (e.g., red box)
        current_ee_pos = self._physics.bind(self._arm.eef_site).xpos.copy()
        self._prev_distance = self._current_distance
        self._current_distance = np.linalg.norm(current_ee_pos - self._target_location)

        # Check if we've reached the target
        goal_threshold = 0.02  # 2cm threshold
        self._goal_achieved = self._current_distance < goal_threshold

        ############################ Reward Computation
        # 1. Reward for decreasing distance to target
        distance_improvement = self._prev_distance - self._current_distance
        distance_reward = 10.0 * distance_improvement

        # 2. Bonus for reaching target
        goal_reward = 100.0 if self._goal_achieved else 0.0

        # 3. Small penalty for excessive movement (encourage smooth trajectories)
        action_penalty = -0.1 * np.sum(np.square(action))

        # Total reward
        reward = distance_reward + goal_reward + action_penalty
        ############################

        # Termination conditions
        terminated = False

        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()

        # Render frame if in human mode
        if self._render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self) -> np.ndarray:
        """
        Renders the current frame and returns it as an RGB array if the render mode is set to "rgb_array".

        Returns:
            np.ndarray: RGB array of the current frame.
        """
        if self._render_mode == "rgb_array" and self._current_frame is not None:
            return self._current_frame
        return None

    def _render_frame(self) -> None:
        """
        Renders the current frame using OpenCV if the render mode is set to "human".
        """
        if self._render_mode != "human" or self._current_frame is None:
            return

        # Initialize step timer if needed
        if self._step_start is None:
            self._step_start = time.time()

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(self._current_frame, cv2.COLOR_RGB2BGR)

        # Display the frame
        cv2.imshow("Widow Arm Simulation - Front View", frame_bgr)
        cv2.waitKey(1)  # Required for OpenCV to update the window

        # Maintain consistent frame rate
        time_until_next_step = self._timestep - (time.time() - self._step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        self._step_start = time.time()

    def close(self) -> None:
        """
        Closes the OpenCV window if it's open.
        """
        if self._cv_window_initialized:
            cv2.destroyWindow("Widow Arm Simulation - Front View")
            self._cv_window_initialized = False
