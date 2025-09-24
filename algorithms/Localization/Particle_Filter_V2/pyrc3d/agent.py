import numpy as np
import pybullet as p
from utilities import utils

# TODO:
# - create car constraints from yaml since it's to jargon.

class Car():
    """
    A module navigating a car model in the sim env.
    """
    def __init__(self, 
            urdf_path: str,
            car_config: str,
            sensors=None,
            debug=False
        ) -> None:
        """
        Initiallize Car object and unpack user's configs.

        Args:
            urdf_path:  Path to the car's urdf file.
            car_config: Path to the user's car config file.
            debug:      If true, turn on sliders for manual control.
        Returns:
            None
        Raises:
            None.
        """

        # Some constants.
        self.URDF_PATH = urdf_path
        self.DEBUG = debug

        # ----------------------------------------------------------
        # Extract car settings from user's config file.
        configs = self.__unpack_car_settings(car_config)

        # Joints' configs.
        self.WHEELS         = configs['wheel_joints']
        self.STEERING       = configs['steering_joints']

        # Velocity, steering, & max force configs.
        vs = configs['velocity']
        ss = configs['steering']
        fs = configs['force']

        self.v_min, self.v_max, self.v_default = vs[0], vs[1], vs[2]
        self.s_min, self.s_max, self.s_default = ss[0], ss[1], ss[2]
        self.f_min, self.f_max, self.f_default = fs[0], fs[1], fs[2]

        # Car's initial position configs.
        self.car_x = configs['car_x']
        self.car_y = configs['car_y']
        # ----------------------------------------------------------

        # Loads the car's urdf file.
        self.car_id = p.loadURDF(self.URDF_PATH)

        # Setup sensors.
        # The `self.sensors` keys are strings of sensor names, for
        # example: 'lidar', 'camera', etc. and the values are an
        # instance of that sensor.
        # -----------------------
        # | Key      | Values   |
        # -----------------------
        # | 'lidar'  | Lidar()  |
        # | 'camera' | Camera() |
        # -----------------------
        if sensors is not None:
            self.sensors = {}
            for sensor in sensors:
                # Get string representation of `sensor`,
                # instantiate the sensor, and call
                # its `setup()` method.
                sensor_name = sensor.__name__.lower()
                self.sensors[sensor_name] = sensor(
                            car_id=self.car_id,
                            configs=configs[sensor_name + '_configs']
                        )
                self.sensors[sensor.__name__.lower()].setup()

    def place_car(self, 
            floor,
            xy_coord=(None, None)
        ) -> None:
        """
        Places the car in the sym env.

        Args:
            floor:      The sim env's floor. This is required
                        to determine the correct z-axis of objects.
            xy_coord:   A tuple (x, y) specifying initial x and y
                        coord. of the car.
        Returns:
            None.
        Raises:
            AssertionError:
            - If `xy_coord` is not a tuple.
                            
        """
        assert isinstance(xy_coord, tuple)

        # Retrieve car's unique id.
        car_id = self.car_id

        # Get (x, y, z) initial coords of the car.
        if xy_coord == (None, None):
            # use (x, y) specified in config file.
            car_x = self.car_x
            car_y = self.car_y
        else:
            # Use (x, y) in `xy_coord`.
            car_x = xy_coord[0]
            car_y = xy_coord[1]
        car_z = utils.stable_z(car_id, floor)
        self.car_z = car_z

        # Place car in the sim env.
        utils.set_point(car_id, [car_x, car_y, car_z])

        # Reset the car joints.
        self.__reset_car_joints(car_id)

        # Asssume differential car for now,
        # so always call this function.
        self.__create_car_constraints(car_id)

        # If debug=True, create sliders.
        if self.DEBUG:
            self.__add_sliders()

    def get_sensor_data(self, sensor: str, common=True):
        """
        Get data from sensor.

        Args:
            sensor: A string specifying the sensor of
                    interest (ex: 'lidar', 'camera')
        Returns:
            The corresponding data of `sensor`.
        Raises:
            None.
        """
        current_car_state = self.get_state(to_array=False, radian=False)
        return self.sensors[sensor].retrieve_data(
                    common=common,
                    car_state=current_car_state
                )
    
    def update_camera_pose(self, x,y,yaw):
        """
        Update camera pose.

        Args:
            x:      x coord.
            y:      y coord.
            yaw:    yaw angle.
        Returns:
            None.
        Raises:
            None.
        """
        print("Updating camera pose...")
        # self.sensors['camera'].translate_camera([x, y, 0])
        # self.sensors['camera'].rotate_camera([0, 0, yaw])


    def get_ctrl_data(self) -> None:
        """
        Get control data.

        Args:
        Returns:
        Raises:
        """
        # TODO
        return None

    def simulate_sensor(self, 
                sensor: str, 
                sensor_data: np.ndarray
            ) -> None:
        """
        Calls `simulate()` method of `sensor`.
        """
        self.sensors[sensor].simulate(sensor_data)

    def navigate(self, algo=None) -> None:
        """
        Given a navigation algorithm `algo`,
        execute `algo` and return the next
        velocity `v`, steering `s`, & max 
        force `f`.

        Args:
            algo: 
            - algo.KalmanFilter()
            - algo.ParticleFilter()
            - ...
        Returns:
            v: Velocity command.
            s: Steering command.
            f: Max force command.
        Raises:
            None
        """
        if self.DEBUG and algo is None:
            v, s, f = self.__exec_manual_control()

        return v, s, f


    def act(self, v: float, s: float, f=20) -> None:
        """
        Take action - i.e: use pybulelt to execute
        changes on the car's joints based on velocity `v`,
        steering `s`, and force `f`.

        Args:
            v: Velocity command.
            s: Steering command.
            f: Max force command.
        Returns:
            None.
        Raises:
            None
        """
        for wheel in self.WHEELS:
            p.setJointMotorControl2(
                        self.car_id, 
                        wheel, 
                        p.VELOCITY_CONTROL, 
                        targetVelocity=v, 
                        force=f
                    )
        	
        for steer in self.STEERING:
            p.setJointMotorControl2(
                        self.car_id, 
                        steer, 
                        p.POSITION_CONTROL, 
                        targetPosition=-s
                    )

    def get_state(self, to_array=True, radian=True) -> np.array:
        """
        Returns the pose of the car where pose is
        given by: (x, y, yaw), where `x` and `y` is
        the car's coordinate in the world's frame, 
        and `yaw` is the car's yaw in its own frame
        (i.e: the car's frame)

        Args:
            radian: If true, return `yaw` angle in radian.
            to_array: If true, return pose as an array. Otherwise,
                      return `x`, `y`, `yaw` individually.
        Returns:
            pose: Pose `x`, `y`, `yaw`of the car as a (1, 3) numpy 
                  array if `to_array` is True, otherwise return
                  them individually.
        """
        # Get car's x,y,z coordinate in world frame.
        coord = utils.get_xyz_point(self.car_id)

        # # Car's yaw in its own coordinate frame.
        # yaw_world_frame = utils.get_yaw(coord)
        # yaw = self.__get_car_yaw(coord, yaw_world=yaw_world_frame)
        # yaw = np.array(yaw).reshape(-1, )
        
        coord = np.array(coord).reshape(-1, )

        # Car's yaw (theta) in world coordinate frame. (Range: [-pi, pi])
        orientation = utils.get_pose(self.car_id)[1]
        yaw = p.getEulerFromQuaternion(orientation)[2]

        if not radian:
            yaw *= round(float(180/np.pi), 2)

        pose = np.hstack([coord[:2], yaw]) # omit z-coord.

        if not to_array:
            # Returns x, y and yaw as individual floats.
            return pose[0], pose[1], pose[2]

        return pose

    def __get_car_yaw(
            self,
            pos,
            roll_world=0,
            pitch_world=0,
            yaw_world=0
        ):
        """
        Get car's yaw in its own coordinate frame

        Default values for roll and pitch are 0 because
        we obviously don't want the car to be rolled or pitched.
        Defautl value for yaw is 0 because we always assume
        it starts by facing the positive x-axis of the world frame.

        Args:
            pos: Car's position in the world frame (x, y, z).
            roll_world: Car's roll in world frame.
            pitch_world: Car's pitch in world frame.
            yaw_world: Car's yaw in world frame.
        Return:
            yaw_local: Car's yaw in car's own frame.
        """
        # Define car's orientation in world frame
        car_orient = p.getQuaternionFromEuler([0, 0, yaw_world])

        # Define world frame orientation in 
        # car's frame (inverse of car_orient)
        inv_car_orient = p.invertTransform([0, 0, 0], car_orient)[1]

        # Multiply car position and orientation by world frame orient. in car frame
        car_pos_local, car_orient_local = p.multiplyTransforms(
                    [0, 0, 0], inv_car_orient, pos, car_orient
                )

        car_euler_local = p.getEulerFromQuaternion(car_orient_local)
        yaw_local = car_euler_local[2]

        return yaw_local


    def __add_sliders(self) -> None:
        """
        Add sliders on the sim env.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        v_min, v_max, v_def = self.v_min, self.v_max, self.v_default
        s_min, s_max, s_def = self.s_min, self.s_max, self.s_default
        f_min, f_max, f_def = self.f_min, self.f_max, self.f_default

        self.v_slider = p.addUserDebugParameter(
                    "Wheels' velocity", 
                    v_min, v_max, v_def
                )
        self.s_slider = p.addUserDebugParameter(
                    "Steering", 
                    s_min, s_max, s_def
                )
        self.f_slider = p.addUserDebugParameter(
                    "Max force",
                    f_min, f_max, f_def
                )

    def __exec_manual_control(self) -> float:
        """
        Manually control the car's states via sliders.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        v = p.readUserDebugParameter(self.v_slider)
        s = p.readUserDebugParameter(self.s_slider)
        f = p.readUserDebugParameter(self.f_slider)
        return v, s, f

    def __reset_car_joints(self, car_id) -> None:
        """
        Reset car's joints positions/states.
        Args:
            car_id: The car's unique id.
        Returns:
            None.
        Raises:
            None.
        """
        for wheel in range(p.getNumJoints(car_id)):
            p.setJointMotorControl2(car_id, wheel,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=0,
                                    force=0)

    def __unpack_car_settings(self, car_config: str) -> dict:
        """
        Unpack environment settings from .yaml
        user config file.
        Args:
            env: A string containig the name of the .yaml
                 file.
        Returns:
            A dictionary containing the following key, value
            pairs:
                TODO!
                path:  path of car's urdf file.
                .
                .
                .
        Raises:
            None.
        """
        configs = utils.load_configs(car_config)
        return configs

    def __create_car_constraints(self, car_id) -> None:
        """
        Create constraints on the car's joints to enable
        differential controls.

        Args:
            car_id: The car's unique id.
        Returns:
            None.
        Raises:
            None.
        """
        # TODO: This function has a lot of repetitive commands.
        # So, consider specifying car constraints in a .yaml file.

        c = p.createConstraint(car_id, 9, car_id, 11, jointType=p.JOINT_GEAR,
                               jointAxis=[0,1,0],
                               parentFramePosition=[0,0,0],
                               childFramePosition=[0,0,0])
        p.changeConstraint(c, gearRatio=1, maxForce=10000)
        
        c = p.createConstraint(car_id, 10, car_id, 13, jointType=p.JOINT_GEAR,
                               jointAxis=[0,1,0],
                               parentFramePosition=[0,0,0],
                               childFramePosition=[0,0,0])
        p.changeConstraint(c, gearRatio=-1, maxForce=10000)
        
        c = p.createConstraint(car_id, 9, car_id, 13, jointType=p.JOINT_GEAR,
                               jointAxis=[0,1,0],
                               parentFramePosition=[0,0,0],
                               childFramePosition=[0,0,0])
        p.changeConstraint(c, gearRatio=-1, maxForce=10000)
        
        c = p.createConstraint(car_id, 16, car_id, 18, jointType=p.JOINT_GEAR,
                               jointAxis=[0,1,0],
                               parentFramePosition=[0,0,0],
                               childFramePosition=[0,0,0])
        p.changeConstraint(c, gearRatio=1, maxForce=10000)
        
        
        c = p.createConstraint(car_id, 16, car_id, 19, jointType=p.JOINT_GEAR,
                               jointAxis=[0,1,0],
                               parentFramePosition=[0,0,0],
                               childFramePosition=[0,0,0])
        p.changeConstraint(c, gearRatio=-1, maxForce=10000)
        
        c = p.createConstraint(car_id, 17,car_id, 19, jointType=p.JOINT_GEAR,
                               jointAxis=[0,1,0],
                               parentFramePosition=[0,0,0],
                               childFramePosition=[0,0,0])
        p.changeConstraint(c, gearRatio=-1, maxForce=10000)
        
        c = p.createConstraint(car_id, 1, car_id, 18, jointType=p.JOINT_GEAR,
                               jointAxis=[0,1,0],
                               parentFramePosition=[0,0,0],
                               childFramePosition=[0,0,0])
        p.changeConstraint(c, gearRatio=-1, gearAuxLink = 15, maxForce=10000)

        c = p.createConstraint(car_id, 3, car_id, 19, jointType=p.JOINT_GEAR,
                               jointAxis=[0,1,0],
                               parentFramePosition=[0,0,0],
                               childFramePosition=[0,0,0])
        p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)
