# ---------------------------------
# Car: Code to get simulator running. (Ackermann Drive)
# ---------------------------------

from matplotlib import pyplot as plt
from rpg.agent import Car
from rpg.simulation import Sim
from rpg.sensors import Camera, Lidar, IMU
from algorithms.Control.Controller import KeyboardController
from utilities.timings import Timings
from OGM import OGM
import numpy as np
import pybullet as p
import math
# Declare user-specific paths to files.
ENV_PATH = "src/configs/env/simple_env.yaml"
CAR_PATH = "src/configs/car/car_config.yaml"
CAR_URDF_PATH = "src/configs/resources/f10_racecar/racecar_ackermann.urdf"

# FPS constants.
CTRL_FPS = 100 # Perform control at 100Hz
LIDAR_FPS = 30 # Simulate lidar at 30Hz
CAMERA_FPS = 30 # Simulate camera at 30Hz
PRINT_FPS = 4 # Print `dist` every 5 seconds for debugging

# Declare sensors.
SENSORS = [Camera, Lidar, IMU]

# Declare controller
controller = KeyboardController(ctrl_fps=CTRL_FPS)

if __name__ == "__main__":
    
    # Initialize simulation.
    sim = Sim(time_step_freq=120, debug=False)
    sim.create_env(env_config=ENV_PATH) # add extra argument to load file with map

    car = Car(
            urdf_path=CAR_URDF_PATH, 
            car_config=CAR_PATH, 
            sensors=SENSORS,
            debug=False
    )

    car.place_car(sim.floor)

    # Set sim response time.
    ctrl_time = Timings(CTRL_FPS)
    camera_time = Timings(CAMERA_FPS)
    print_frequency = Timings(PRINT_FPS)

    pose = None # Car's state (in this case, it's the car's pose)
    
    # creating matplotlib map for the og    

        
    
    
    
    
    
    
    
    ogm=OGM()
    ogm.showPlots()
    while True:
        try:
            pose = car.get_state(to_array=False,radian=True)
            x, y, yaw = pose
            y=-y

            ## ... Algorithm Inserted Here (e.g. PID Control) ... ##

            rays_data, dists, coords = car.get_sensor_data("lidar")
            car.simulate_sensor("lidar", rays_data)

            imu_data = car.get_sensor_data("imu")
            imu_lin_accel = imu_data["linear_acceleration"]
            imu_ang_vel = imu_data["angular_velocity"]
            
            ogm.bressenham_mark_Cells(np.array(dists), np.array([x,y,yaw]))
            # x2,y2=ogm.meter_to_cell(np.array([x,y]))
            # ogm.ogm_plot(x2,y2,True)
            

            if print_frequency.update_time():
                robot_pose=np.array([x,y,yaw])
                sensor_pose = robot_pose + np.array([np.cos(robot_pose[2])*0.265 - np.sin(robot_pose[2])*0, np.sin(robot_pose[2])*0.265 + np.cos(robot_pose[2])*0, -math.pi/2])
                pass # debugging statements

            if ctrl_time.update_time():
                v, s = controller.navigate(x, y, yaw)    
            
                car.act(v, s)
                
                sim.step() # Advance one time step in the sim
                sim.view(x,-y, yaw, "distant")
        except KeyboardInterrupt:
            print("\n[INFO] Ctrl+C detected, stopping simulation...")
            break
            
    print("-"*100)
    ogm.updatePlot()
    print("updated the plot properly")
    ogm.showPlots()  # block here so window stays open
    plt.pause(100000)
    print("here")