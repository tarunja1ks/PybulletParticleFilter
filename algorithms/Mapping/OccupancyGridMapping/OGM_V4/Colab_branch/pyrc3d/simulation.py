# TODO:
# - Config for colors (like wall, obstacles and floor color)

import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from IPython.display import clear_output
from repository.utilities import utils, geometry_utils
import time

class Sim():
    """
    Class to manage simulation environment.
    """
    def __init__(self, 
            debug=False,
            gravity=-9.8,
            time_step_freq=240
        ) -> None:
        """
        Initiallize Sim object.

        Args:
            debug:          If True, activate debugger window.
            gravity:        Set the gravity value of the simulation.
            time_step_freq: The frequency when calling step().
        Returns:
            None.
        Raises:
            None.
        """
        self.CLIENT = None
        self.GRAVITY = gravity
        self.TIME_STEP = time_step_freq # Hz

        # To place agent (car) when calling car.place_car()
        self.floor = None 

        # If not debugging, remove debugger window
        self.DEBUG = debug

    def create_env(self, 
            env_config: str, 
            agent_pos=(0, 0),
            goal_loc=None,
            custom_env_path=None,
            GUI=True
        ) -> None:
        """
        Create an environment according to
        a .yaml specified by the arg `env_config`.

        Args:
            env_config: A .yaml config file specifying
                        the environment.
            custom_env_path: A path to a `.txt` file specifying
                             custom environment.
            GUI: If True, use GUI mode, else use
                 DIRECT mode. Set to False when 
                 runnning on servers without GPUs.
        Returns:
            None.
        Raises:
            None.
        """
        self.env_config = env_config
        self.agent_pos = agent_pos

        if custom_env_path is not None:
            self.custom_env_path = custom_env_path

        self.GUI = GUI
        if GUI:
            self.CLIENT = p.connect(p.GUI)
        else:
            self.CLIENT = p.connect(p.DIRECT)
        pass

        if not self.DEBUG:
            self.__remove_debugger_window()

        # Reset the env.
        self.reset()

        # Set gravity of the env.
        p.setGravity(0, 0, self.GRAVITY)

        # Set time step of env when `step` is called.
        p.setTimeStep(1./self.TIME_STEP)

        # Unpacking sim config.
        sim_config = self.__unpack_env_settings()

        self.custom      = sim_config['custom']
        self.wall_w      = sim_config['wall_w']
        self.floor_s     = sim_config['floor_s']
        self.floor_h     = sim_config['floor_h']
        self.n_obs       = sim_config['n_obs']
        self.obs_h       = sim_config['obs_h']
        self.obs_w       = sim_config['obs_w']
        self.goal_w      = sim_config['goal_w']
        self.goal_h      = sim_config['goal_h']
        self.wall_color  = sim_config['wall_color']
        self.obs_color   = sim_config['obs_color']
        self.goal_color  = sim_config['goal_color']
        self.floor_color = sim_config['floor_color']
        self.goal_loc    = sim_config['goal_loc']

        # Init. color attributes
        self.__create_colors()

        # Set the walls and floors.
        self.__set_walls_and_floors()

        # Place obstacles.
        if not self.custom:
            self.__place_obstacles()
        else:
            self.__place_obstacles(custom=self.custom)
        
        # Place the goal.
        self.__place_goal(loc=goal_loc)

        self.fig, self.ax = plt.subplots()

    def step(self) -> None:
        """
        Advance one time step in the simulation.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        p.stepSimulation(physicsClientId=self.CLIENT)

    def get_env_info(self) -> dict:
        """
        Returns a dictionary containing relevant
        information of the sim env. Users may find this
        method very useful when implementing navigation
        algorithms.
        """
        sim_info = {
            "num_obstacles": self.n_obs,
            "map_size": self.floor_s
        }
        return sim_info

    def reset(self) -> None:
        """
        Reset the simulation, i.e: remove all objects
        from the env., restore it to the initial
        conditions, and call create_env again.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        p.resetSimulation(physicsClientId=self.CLIENT)

    def kill_env(self) -> None:
        """
        Disconnect from the physics server.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        p.disconnect(physicsClientId=self.CLIENT)

    def image_env(self) -> None:
        """
        Display the environment as an image, for use in Google Colab notebooks.
        """

        DISTANCE = 20

        # Get the camera view matrix
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0],
            distance=DISTANCE,
            yaw=90,
            pitch=-90,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self.CLIENT
        )

        # Get the camera projection matrix
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.1,
            farVal=100.0,
            physicsClientId=self.CLIENT
        )

        # Reset the camera pose to view the entire map
        p.resetDebugVisualizerCamera(
            cameraDistance=DISTANCE,
            cameraYaw=90,
            cameraPitch=-90,
            cameraTargetPosition=[0, 0, 0],
            physicsClientId=self.CLIENT
        )

        # Get the image from the camera
        (_, _, px, _, _) = p.getCameraImage(
            width=224,
            height=224,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.CLIENT
        )

        # Convert RGB array to image
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (224, 224, 4))
        rgb_array = rgb_array[:, :, :3]

        # Display the image
        time.sleep(0.05)
        clear_output(wait=True)
        plt.imshow(rgb_array)
        # Find average color of the image
        avg_color = np.mean(rgb_array, axis=(0, 1))
        print(avg_color)
        plt.show()


    def __place_goal(self, loc=None):
        goal = utils.create_box(
                    self.goal_w, 
                    self.goal_w, 
                    self.goal_h, 
                    color=self.goal_color
                )

        if loc is None:
            loc = self.__select_random_locs(
                        1,
                        self.goal_w,
                    )

            while self.__goal_overlaps_with_obstacles(loc[0]):
                loc = self.__select_random_locs(
                            1,
                            self.goal_w,
                        )
            loc = loc[0]

        x, y = loc
        self.goal_loc = (x, y)
        utils.set_point(goal, [x, y, self.goal_h/2.])

    def __goal_overlaps_with_obstacles(self, goal_loc):
        for obs_coord in self.obstacle_coordinates:
            x_obs, y_obs = obs_coord
            x_goal, y_goal = goal_loc
            if abs(x_goal - x_obs) < self.goal_w and abs(y_goal - y_obs) < self.goal_w:
                return True

        return False

    def __set_walls_and_floors(self) -> None: 
        """
        TODO: Decription.
        """
        # ---------------------------------------------------------------------------
        # Create floor.
        floor = utils.create_box(self.floor_s, self.floor_s, self.floor_h, color=self.floor_color)
        # Adding frictions to floor for the car.
        p.changeDynamics(floor, -1,
                         lateralFriction=1.0,
                         rollingFriction=0.01,
                         spinningFriction=0.0)
        # Keep floor so can place car.
        self.floor = floor
        utils.set_point(floor, geometry_utils.Point(z=-self.floor_h/2.))
        # ---------------------------------------------------------------------------

        wall1 = utils.create_box(
                    self.floor_s + self.wall_w, 
                    self.wall_w, self.wall_w, 
                    color=self.wall_color
                )
        utils.set_point(wall1, geometry_utils.Point(y=self.floor_s/2., z=self.wall_w/2.))

        wall2 = utils.create_box(
                    self.floor_s + self.wall_w, 
                    self.wall_w, self.wall_w, 
                    color=self.wall_color
                )
        utils.set_point(wall2, geometry_utils.Point(y=-self.floor_s/2., z=self.wall_w/2.))

        wall3 = utils.create_box(
                    self.wall_w, 
                    self.floor_s + self.wall_w, 
                    self.wall_w, color=self.wall_color
                )
        utils.set_point(wall3, geometry_utils.Point(x=self.floor_s/2., z=self.wall_w/2.))

        wall4 = utils.create_box(
                    self.wall_w, 
                    self.floor_s + self.wall_w, 
                    self.wall_w, color=self.wall_color
                )
        utils.set_point(wall4, geometry_utils.Point(x=-self.floor_s/2., z=self.wall_w/2.))

    def __place_obstacles(self, custom=False) -> None:
        """
        TODO: Decription.
        """
        if not self.custom:
            obstacles = []
            for _ in range(self.n_obs):
                body = utils.create_box(self.obs_w, self.obs_w, self.obs_h, color=self.obs_color)
                obstacles.append(body)

            locs = self.__select_random_locs(self.n_obs, self.obs_w)
            self.obstacle_coordinates = locs
            for obst, coords in zip(obstacles, locs):
                x = coords[0]
                y = coords[1]
                utils.set_point(obst, [x, y, self.obs_h/2.])
        else:
            obstacles_info = self.__read_custom_env_file(self.custom_env_path)
            obstacles = []
            for i in range(obstacles_info.shape[0]):
                obs_w = obstacles_info[i, 2]
                obs_l = obstacles_info[i, 3]
                body = utils.create_box(obs_w, obs_l, self.obs_h, color=self.obs_color)
                obstacles.append(body)
            coords = obstacles_info[:, :2]
            for obst, coords in zip(obstacles, coords):
                x = coords[0]
                y = coords[1]
                utils.set_point(obst, [x, y, self.obs_h/2.])
            self.obstacle_coordinates = coords.reshape(-1, 2).tolist()

    def __select_random_locs(self, N, w):
        """
        Generate `N` random coordinates in the sim env for
        objects of widh `w` such that none of them will
        overlap.
        Args:
            N: Number of coordinates to generate.
            w: Width of the object whose coordinates we're 
               generating.
        Returns:
            centers: The coordinate of the centers of the `N`
                     objects width width `w`.
        Raises:
            None.
        """
        # Define the range of possible coordinates for the obstacle centers
        valid_size = self.floor_s - self.wall_w # need to consider wall width
        x_range = (w/2 - valid_size/2, valid_size/2 - w/2)
        y_range = (w/2 - valid_size/2, valid_size/2 - w/2)
        
        # Initialize an empty list of obstacle centers
        centers = []

        # Generate n random obstacle centers until we have enough non-overlapping ones
        while len(centers) < N:
            # Generate a random center
            center = (round(random.uniform(*x_range), 1), round(random.uniform(*y_range), 1))
            # Check if the obstacle overlaps with any of the previous ones
            overlaps = False
            for other in centers:
                if abs(center[0] - other[0]) < w and abs(center[1] - other[1]) < w:
                    overlaps = True
                    break

            # Check if the obstacle overlaps with starting coord or the area covered by width w at starting coord.

            if abs(center[0] - self.agent_pos[0]) < w and abs(center[1] - self.agent_pos[1]) < w:
                overlaps = True
            
            # If the obstacle doesn't overlap, add it to the list of centers
            if not overlaps:
                centers.append(center)

        return centers

    def __read_custom_env_file(self, filepath):
        # Open file and read contents
        with open(filepath, 'r') as f:
            contents = f.readlines()
    
        # Create numpy array to hold data
        data = np.zeros((len(contents), 4))
    
        # Loop through each line and add to numpy array
        for i, line in enumerate(contents):
            values = line.strip().split(',')
            data[i] = [float(v) for v in values]
    
        return data

    def __create_colors(self):
        self.wall_color = utils.hex_to_rgba(self.wall_color)
        self.obs_color = utils.hex_to_rgba(self.obs_color)
        self.goal_color = utils.hex_to_rgba(self.goal_color)
        self.floor_color = utils.hex_to_rgba(self.floor_color)

    def __unpack_env_settings(self) -> dict:
        """
        Unpack environment settings from .yaml
        user config file.

        Returns:
            A dictionary containing the following key, value
            pairs:
                wall_w:  width of side walls.
                floor_s: Size of floor.
                floor_h: Floor height.
                n_obs:   Number of obstacles in the sim.
                obs_w:   Width of obstacles.
                obs_h:   Height of obstacles.
        Raises:
            None.
        """
        configs = utils.load_configs(self.env_config)
        return configs

    def __remove_debugger_window(self):
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.CLIENT)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=self.CLIENT)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=self.CLIENT)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=self.CLIENT)
