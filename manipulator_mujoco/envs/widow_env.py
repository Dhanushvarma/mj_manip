import time
import numpy as np
import os
import cv2
import mujoco.viewer
from loguru import logger
from dm_control import mjcf
from dm_control.manipulation.shared.constants import RED, GREEN
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

    def __init__(self, render_mode=None, render_backend="mjviewer"):
        """
        Initialize the WidowEnv environment.

        Args:
            render_mode (str, optional): Rendering mode, either "human" or "rgb_array".
            render_backend (str, optional): Rendering backend, either "cv2" or "mjviewer".
        """
        # Validate render_backend
        assert render_backend in [
            "cv2",
            "mjviewer",
        ], "render_backend must be either 'cv2' or 'mjviewer'"
        self._render_backend = render_backend

        # set all env relevant constants
        self._init_consts()

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
                    low=0,
                    high=255,
                    shape=(self._image_height, self._image_width, 3),
                    dtype=np.uint8,
                ),  # RGB image with useful dimensions
                "topview_image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self._image_height, self._image_width, 3),
                    dtype=np.uint8,
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

        # table
        self._table = Primitive(
            type="box",
            size=self._table_dims,
            pos=[0, 0, self._table_dims[-1]],
            rgba=[1, 1, 0, 1],
            friction=[1, 0.3, 0.0001],
        )

        # red box
        self._red_box = Primitive(
            type="box",
            size=[self._cube_prop_size] * 3,
            pos=[0, 0, self._cube_prop_size],
            rgba=RED,
            friction=[1, 0.3, 0.0001],
            mass=0.01,
        )
        self._green_box = Primitive(
            type="box",
            size=[self._cube_prop_size] * 3,
            pos=[0, 0, self._cube_prop_size],
            rgba=GREEN,
            friction=[1, 0.3, 0.0001],
            mass=0.01,
        )

        # attach arm, table, boxes
        self._arena.attach(self._arm.mjcf_model, pos=self._robot_root_pose)
        self._arena.attach(self._table.mjcf_model, pos=self._table_root_pose)
        self._arena.attach_free(
            self._red_box.mjcf_model,
            pos=[
                self._table_root_pose[0],
                self._cube_y_offset,
                2 * self._table_dims[2],
            ],
        )
        self._arena.attach_free(
            self._green_box.mjcf_model,
            pos=[
                self._table_root_pose[0],
                -self._cube_y_offset,
                2 * self._table_dims[2],
            ],
        )

        # attach cameras
        self._arena.attach_camera(
            name="frontview", pos=[1.15, 0, 0.2125], quat=[0.5, 0.5, 0.5, 0.5]
        )
        self._arena.attach_camera(
            name="topview", pos=[0.38, 0, 0.8], quat=[0.7071068, 0, 0, 0.7071068]
        )

        # generate model
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        # increase camera buffer size
        self._physics.model.vis.global_.offwidth = self._image_width
        self._physics.model.vis.global_.offheight = self._image_height

        # set up OSC controller with appropriate parameters for Widow Arm
        self._controller = OperationalSpaceController(
            physics=self._physics,
            joints=self._arm.joints,
            eef_site=self._arm.eef_site,
            min_effort=-25.0,
            max_effort=25.0,
            kp=self._controller_gains["kp"],
            ko=self._controller_gains["ko"],
            kv=self._controller_gains["kv"],
            vmax_xyz=1.0,
            vmax_abg=1.0,
        )

        # for time keeping
        self._timestep = self._physics.model.opt.timestep
        self._step_start = None

        # Initialize rendering backends
        self._cv_window_initialized = False
        self._viewer = None

        # Initialize the selected rendering backend
        if self._render_mode == "human":
            if self._render_backend == "cv2":
                cv2.namedWindow("Widow Arm Simulation - Front View", cv2.WINDOW_NORMAL)
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

    def _init_consts(self):
        """
        All constant relevant to setting up the task
        """
        self._controller_gains = {"kv": 50, "kp": 200, "ko": 200}
        self._robot_rest_joint_cfg = [0, 0, 0, 0, 0, 0, 0.015, -0.015]

        self._image_width = 1920
        self._image_height = 1080

        self._table_dims = [0.15, 0.3, 0.1]
        self._cube_prop_size = 0.02

        self._robot_root_pose = [0, 0, 0]
        self._table_root_pose = [0.375, 0, 0]

        self._cube_y_offset = 0.15
        self._box_spawn_bounds = [0.15, 0.15, 0]  # xyz from root pose

        self._height_off_table = 0.1

    def _get_obs(self) -> dict:
        # Get pinch site pose (position and orientation)
        pinch_site_pos = self._physics.bind(self._arm.eef_site).xpos.copy()
        pinch_site_quat = self._physics.bind(self._arm.eef_site).quat.copy()
        pinch_site_pose = np.concatenate([pinch_site_pos, pinch_site_quat])

        # Get joint positions
        joint_pose = self._physics.bind(self._arm.joints).qpos.copy()

        # Get camera images based on the rendering backend
        if self._render_backend == "mjviewer" and self._render_mode == "human":
            # Use zeros trick to avoid conflict with MuJoCo viewer
            frontview_image = np.zeros(
                (self._image_height, self._image_width, 3), dtype=np.uint8
            )
            topview_image = np.zeros(
                (self._image_height, self._image_width, 3), dtype=np.uint8
            )
        else:
            # Render actual camera images
            frontview_image = self._physics.render(
                height=self._image_height,
                width=self._image_width,
                camera_id="frontview",
            )
            topview_image = self._physics.render(
                height=self._image_height, width=self._image_width, camera_id="topview"
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
        return info

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)

        print("Reset is occurring!")

        # Use the seed for randomization
        self.np_random = np.random.RandomState(seed)

        # reset physics
        with self._physics.reset_context():
            # put arm in a reasonable starting position
            self._physics.bind(self._arm.joints).qpos = self._robot_rest_joint_cfg

            # randomize box positions
            red_box_pos = [
                self.np_random.uniform(
                    self._table_root_pose[0] - self._box_spawn_bounds[0],
                    self._table_root_pose[0] + self._box_spawn_bounds[0],
                ),
                self.np_random.uniform(
                    self._cube_y_offset - self._box_spawn_bounds[1],
                    self._cube_y_offset + self._box_spawn_bounds[1],
                ),
                2 * self._table_dims[2],
            ]
            green_box_pos = [
                self.np_random.uniform(
                    self._table_root_pose[0] - self._box_spawn_bounds[0],
                    self._table_root_pose[0] + self._box_spawn_bounds[0],
                ),
                self.np_random.uniform(
                    -self._cube_y_offset - self._box_spawn_bounds[1],
                    -self._cube_y_offset + self._box_spawn_bounds[1],
                ),
                2 * self._table_dims[2],
            ]
            self._physics.bind(self._red_box.geom).xpos = red_box_pos
            self._physics.bind(self._green_box.geom).xpos = green_box_pos

            # TODO: change to fk solution of the arm reset joint configuration
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

        # Render the initial state if in human mode
        if self._render_mode == "human":
            self._render_frame()

        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> tuple:
        # hack: to access grip
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
        Renders the current frame using the selected backend if the render mode is set to "human".
        """
        if self._render_mode != "human":
            return

        # Initialize step timer if needed
        if self._step_start is None:
            self._step_start = time.time()

        # Render using the selected backend
        if self._render_backend == "cv2":
            if self._current_frame is None:
                return

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(self._current_frame, cv2.COLOR_RGB2BGR)

            # Display the frame
            cv2.imshow("Widow Arm Simulation - Front View", frame_bgr)
            cv2.waitKey(1)  # Required for OpenCV to update the window

        elif self._render_backend == "mjviewer":
            if self._viewer is None:
                # Launch MuJoCo viewer
                self._viewer = mujoco.viewer.launch_passive(
                    self._physics.model.ptr,
                    self._physics.data.ptr,
                )

            # Render viewer
            self._viewer.sync()

        # Maintain consistent frame rate
        time_until_next_step = self._timestep - (time.time() - self._step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        self._step_start = time.time()

    def close(self) -> None:
        """
        Closes the rendering backend.
        """
        if self._render_backend == "cv2" and self._cv_window_initialized:
            cv2.destroyWindow("Widow Arm Simulation - Front View")
            self._cv_window_initialized = False
        elif self._render_backend == "mjviewer" and self._viewer is not None:
            self._viewer.close()
            self._viewer = None
