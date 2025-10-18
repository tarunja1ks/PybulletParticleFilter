# ---------------------------------
# husky_kuka: Code to get simulator running. (Ackermann Drive)
# ---------------------------------

from matplotlib import pyplot as plt
from rpg.agent import HuskyKuka
from rpg.simulation import Sim
from rpg.sensors import Camera, Lidar, IMU
from algorithms.Control.Controller import KeyboardController
from utilities.timings import Timings
from OGM import OGM
from ParticleFilter import ParticleFilter
import numpy as np
import pybullet as p
import math
import time


# Declare user-specific paths to files.
ENV_PATH = "src/configs/env/simple_env.yaml"
HUSKY_PATH = "src/configs/husky_kuka/husky_kuka_config.yaml"
KUKA_URDF_PATH = "src/configs/resources/kuka_iiwa/model_free_base.urdf"
HUSKY_URDF_PATH = "src/configs/resources/husky/husky.urdf"

# FPS constants.
CTRL_FPS = 100 # Perform control at 100Hz
LIDAR_FPS = 30 # Simulate lidar at 30Hz
CAMERA_FPS = 30 # Simulate camera at 30Hz
PRINT_FPS = 4 # Print `dist` every 5 seconds for debugging

# Declare sensors.
SENSORS = [Camera, Lidar, IMU]

# Declare controller

controller = KeyboardController(ctrl_fps=CTRL_FPS, 
                                arm=False, 
                                forward=True, 
                                num_joints=7) # Control the husky_kuka and arm with the keyboard

if __name__ == "__main__":
    
    # Initialize simulation.
    sim = Sim(time_step_freq=120, debug=False)
    sim.create_env(env_config=ENV_PATH) # add extra argument to load file with map

    # creating the Husky Differential Drive Robot object
    husky_kuka = HuskyKuka(
        husky_urdf_path=HUSKY_URDF_PATH, 
        kuka_urdf_path=KUKA_URDF_PATH, 
        husky_config=HUSKY_PATH,
        sensors=SENSORS,
        debug=False
        )

    husky_kuka.place_robot(floor=sim.floor) # loading the Husky Robot into the environment

    # Set sim response time.
    ctrl_time = Timings(CTRL_FPS)
    camera_time = Timings(CAMERA_FPS)
    print_frequency = Timings(PRINT_FPS)

    pose = None # husky_kuka's state (in this case, it's the husky_kuka's pose)
    
    # creating matplotlib map for the og   
    
    
    # ogm
    ogm=OGM()
    
    # particle filter
    numberOfParticles=100 # number of particles
    pf=ParticleFilter(np.array([0,0,0]),ogm,numberOfParticles) # initializing the particle filter
    
    
    simdex=0# tracking the iteration index while moving through the simulation
    
    estimated_pose=[0,0,0] # a pose to start out with for the initial position of the robot

    while True:
        try:
            pose = husky_kuka.get_state(to_array=False, radian=False) # getting the Truth(the actual pose)
            x, y, yaw = pose
            y=-y

            
            # collecting sensor data
            
            rays_data, dists, coords = husky_kuka.get_sensor_data("lidar")
            husky_kuka.simulate_sensor("lidar", rays_data)    
            imu_data=husky_kuka.get_sensor_data("imu")
            imu_lin_vel=imu_data["linear_velocity"][:2]
            imu_ang_vel=imu_data["angular_velocity"]
            dt=imu_data["dt"]
            ya=imu_data["yaw"]
            
            # a debugging if-check to avoid printing every simulation where its unreadable
            if print_frequency.update_time():
                robot_pose=np.array([x,y,yaw])
                print(estimated_pose,"-",robot_pose,"-")
                pass # debugging statements
            
            
            # every 40 iterations in simulation update the visualization of the map
            if(simdex%40==0):
                ogm.show_cv2()
                
                
            simdex+=1 # increasing simulation iteration index
            
            if ctrl_time.update_time():
                # controlling with keyboard
                v, s = controller.navigate(x, y, yaw)    
                
                x,y,yaw*np.pi/180
                e_x,e_y,e_yaw=estimated_pose
                
                # updating the map with lidar data
                ogm.bressenham_mark_Cells(np.array(dists), np.array([e_x,e_y,e_yaw*np.pi/180]))

                
                # performing particle filter to track and estimate pose for robot
                pf.prediction_step(np.tile(imu_lin_vel, (pf.numberofparticles, 1)), np.tile(imu_ang_vel, (pf.numberofparticles, 1)), dt)            
                estimated_pose=pf.update_step(ogm,np.array(dists),40)
                
                # perform keyboard inputs onto the robot
                husky_kuka.act(v, s)
                
                sim.step() # Advance one time step in the sim
                sim.view(x,-y, yaw, "distant")  
                
                
        except KeyboardInterrupt:
            print("\n[INFO] Ctrl+C detected, stopping simulation...")
            break
    print("updated the plot properly")
