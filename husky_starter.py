# ---------------------------------
# Husky + Arm: Code to get simulator running. (Differential Drive)
# ---------------------------------

from rpg.agent import HuskyKuka
from rpg.simulation import Sim
from rpg.sensors import Camera, Lidar, IMU
from algorithms.Control.Controller import KeyboardController
from utilities.timings import Timings

# Declare user-specific paths to files.
ENV_PATH = "src/configs/env/simple_env.yaml"
HUSKY_PATH = "src/configs/husky_kuka/husky_kuka_config.yaml"
KUKA_URDF_PATH = "src/configs/resources/kuka_iiwa/model_free_base.urdf"
HUSKY_URDF_PATH = "src/configs/resources/husky/husky.urdf"

# FPS constants.
CTRL_FPS = 100 # Perform control at 100Hz
LIDAR_FPS = 30 # Simulate lidar at 30Hz
CAMERA_FPS = 30 # Simulate camera at 30Hz
PRINT_FPS = 0.2 # Print `dist` every 5 seconds for debugging

# Declare sensors.
SENSORS = [Camera, Lidar, IMU]

# Declare controller
controller = KeyboardController(ctrl_fps=CTRL_FPS, 
                                arm=True, 
                                forward=True, 
                                num_joints=7) # Control the car and arm with the keyboard

if __name__ == "__main__":
    
    # Initialize simulation.
    sim = Sim(time_step_freq=120, debug=False)
    sim.create_env(env_config=ENV_PATH) # add extra argument to load file with map

    husky_kuka = HuskyKuka(
        husky_urdf_path=HUSKY_URDF_PATH, 
        kuka_urdf_path=KUKA_URDF_PATH, 
        husky_config=HUSKY_PATH,
        sensors=SENSORS,
        debug=False
        )

    husky_kuka.place_robot(floor=sim.floor)

    # Set sim response time.
    ctrl_time = Timings(CTRL_FPS)
    camera_time = Timings(CAMERA_FPS)
    print_frequency = Timings(PRINT_FPS)

    pose = None # Car's state (in this case, it's the car's pose)
        
    while True:
        pose = husky_kuka.get_state(to_array=False, radian=False)
        x, y, yaw = pose

        ## ... Algorithm Inserted Here (e.g. PID Control) ... ##

        rays_data, dists, coords = husky_kuka.get_sensor_data("lidar")
        husky_kuka.simulate_sensor("lidar", rays_data)

        imu_data = husky_kuka.get_sensor_data("imu")
        imu_lin_accel, imu_ang_vel = imu_data["linear_acceleration"], imu_data["angular_velocity"]
        # Print x,y,z lin acceleration rounded to 2 dp
        print(f"Linear Acceleration: {imu_lin_accel[0]:.2f}, {imu_lin_accel[1]:.2f}, {imu_lin_accel[2]:.2f}")

        if print_frequency.update_time():
            pass # debugging statements

        if ctrl_time.update_time():
            v, s, arm_angles = controller.navigate(x, y, yaw) 
            
            # Control the robotic arm
            husky_kuka.control_arm_forward(arm_angles)    
           
            husky_kuka.act(v, s) 
            
            sim.step()
            sim.view(x,y, yaw, "distant")