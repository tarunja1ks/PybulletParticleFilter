# ------------------------------------------------------------------
# 2D Particle Filter Implementation with PyBullet
# (PF2D) | Status: COMPLETED (Version 2)
#        | Contributors: Jeffrey
#
# Key assumptions in PF:
# - Robot positions (states) are known
# - Landmark positions are known
# - There is sensor and motion noise.
# - Particles are initially generated with uniform distribution.
# - The world is static (obstacles are assumed to be not moving)
#
# Problem setting:
#     Given measurement data z, robot state x, and control input u,
#     find the estimated state bel(x_t)
# ------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import cv2
from IPython.display import clear_output
from math import *
from time import time
from Path_Sim import Simulation
from util import World2Grid
from utilities.timings import Timings


class Map():
    """
    A class used to initiate an environment with a given grid map.

    Attributes:
        map (ndarray): Given grid map.
        landmarks (list): Positions of landmarks in this map.
        realMapSize (float): Side length of the map.
        gridMapSize (int): Number of grids per side length.
        res (float): Grid map resolution.
    """

    def __init__(self, map_file: str, realMapSize, res=0.1):
        """
        Constructor of Map to initialize the environment.

        Parameters:
            map_file (str): File name of the input map.
            realMapSize (float): Side length of the map.
            res (float): Grid map resolution.
        """

        self.map = np.loadtxt(map_file, dtype=float)
        self.landmarks = [(2.5/2., 2.5/2.),   (-2.5/2., 2.5/2.),
                          (-2.5/2., -2.5/2.), (2.5/2., -2.5/2.)]
        scale_idx = 3
        self.map = cv2.resize(self.map,
                              dsize=(50*scale_idx, 50*scale_idx),
                              interpolation=cv2.INTER_NEAREST)
        res /= scale_idx
        self.realMapSize = realMapSize
        self.gridMapSize = int(self.realMapSize/res)
        self.res = res

    def World2Grid(self, point):
        """
        The function to transfer the world coordinate of a point to its
            corresponding grid map coordinate.

        Parameters:
            point (tuple): World coordinate of the point.
        
        Returns:
            tuple: Grid map coordinate of the grid.
        """

        return World2Grid(point, self.realMapSize, self.gridMapSize, self.res)

class ParticleFilter():
    """
    A class used to implement PF algorithm and provide visualization.
    """

    class Particle():
        """
        A class used to define each particle.
        """
        def __init__(self, x, y, yaw, weight=0):
            self.pos = np.array([x, y])
            self.yaw = yaw
            self.weight = weight

    class Robot():
        """
        A class used to define the robot.
        """
        def __init__(self, robot_pose):
            x, y, yaw = robot_pose
            self.pos = np.array([x, y])
            self.yaw = yaw

    def __init__(self, input_map: Map, num_particles: int, robot_pose):
        ''' 
        The constructor of ParticleFilter that initializes necessary parameters
            and the robot on the map, while updating the robot and particle
            states and weights based on a pre-defined path.

        Parameters:
            input_map (Map): Map object that stores the information of the map.
            num_particles (int): Number of particles used in this implementation.
            robot_pose (tuple): Initial values of X, Y, Yaw of the robot.
        '''

        self.map = input_map

        # Initialize robot
        self.robot = self.Robot(robot_pose)

        # motion and measurement noise
        self.motion_noise = 0.02
        self.turn_noise = 0.02
        self.bearing_noise = 0.02
        self.distance_noise = 0.5

        # position of landmarks
        self.landmarks = np.array(self.map.landmarks)

        # Initialize particles
        self.num_particles = num_particles
        self.particles = []
        self.weights = np.ones(self.num_particles) / self.num_particles

        weight = 1 / self.num_particles
        Xs = np.random.uniform(-2.45, 2.45, self.num_particles)
        Ys = np.random.uniform(-2.45, 2.45, self.num_particles)
        yaws = np.random.uniform(0, 2*pi, self.num_particles)
        self.particles = np.array([self.Particle(Xs[i], Ys[i], yaws[i], weight)
                                   for i in range(self.num_particles)])
        
    def prediction_step(self, new_robot_pose):
        '''
        This method performs the prediction step and changes the state of the
            particles based the latest robot pose.

        Parameters:
            new_robot_pose (tuple): Latest values of X, Y, Yaw of the robot.
        '''

        new_robot_pos = np.array(new_robot_pose[0:2])
        # Find out the change in robot's pose
        dx, dy = new_robot_pos - self.robot.pos
        d_yaw = new_robot_pose[2] - self.robot.yaw
        # Update robot's current pose
        self.robot.pos, self.robot.yaw = new_robot_pos, new_robot_pose[2]

        # Perform control on all particles based on the robot's motion
        turn_noises = np.random.normal(0, self.turn_noise, self.num_particles)
        motion_noises_x = np.random.normal(0, self.motion_noise, self.num_particles)
        motion_noises_y = np.random.normal(0, self.motion_noise, self.num_particles)
        for i in range(self.num_particles):
            particle = self.particles[i]
            particle.pos[0] += dx + motion_noises_x[i]
            particle.pos[1] += dy + motion_noises_y[i]
            particle.yaw = (particle.yaw + pi + d_yaw + turn_noises[i]) % (2*pi) - pi

    def update_step(self):
        '''
        This method performs the update step and updates the weights of all
            particles based on their simulated distance and bearing angle
            measurements to all landmarks.

        Measures:
            - Distance between robot and each landmark with measurement noise (Gaussian)
            - Bearing between robot's orientation and each landmark with measurement noise (Gaussian)
            - Distance between each particle and each landmark
            - Bearing between each particle's orientation and each landmark
            - Joint probability of each particle based on the above measurements
        '''

        def calc_bearing(landmark, target):
            """
            A helper method to calculate the bearing between a landmark and an
                object.
            """

            vec_t2l = np.array(landmark) - target.pos
            theta_l2x = np.arctan2(vec_t2l[1], vec_t2l[0])
            theta_l2y = theta_l2x - pi/2
            if theta_l2x >= pi/2 or theta_l2x < -pi/2:
                theta_l2y %= pi
            bearing = theta_l2y - target.yaw
            if bearing > pi:
                bearing -= 2*pi
            elif bearing < -pi:
                bearing += 2*pi

            return bearing
        
        num_landmarks = self.landmarks.shape[0]

        # Calculate distance and bearing measurements to all landmarks
        robot_pos_pile = np.tile(self.robot.pos, (num_landmarks, 1))
        dists_r2l = np.linalg.norm(robot_pos_pile-self.landmarks, axis=1) \
                  + np.random.normal(0, self.distance_noise, num_landmarks)
        bear_noise = np.random.normal(0, self.bearing_noise, num_landmarks)
        bearing_rnl = np.array([calc_bearing(self.landmarks[i], self.robot) + bear_noise[i]
                                for i in range(num_landmarks)])

        # Calculate each particle's relative distance and bearing measurements to all landmark
        dist_meas = np.array(
            [np.linalg.norm(particle.pos-landmark)
             for particle in self.particles for landmark in self.landmarks]
            ).reshape(self.num_particles, num_landmarks)
        bear_meas = np.array(
            [calc_bearing(landmark, particle)
             for particle in self.particles for landmark in self.landmarks]
            ).reshape(self.num_particles, num_landmarks)
        
        # Calculate the likelihood/weight of each particle based on distance and bearing measurements (joint probability (product) of two 1D Gaussian pdfs)
        denom = 2 * pi * self.distance_noise * self.bearing_noise
        weights = np.array(
            [(np.exp(-((dists_r2l[j]-dist_meas[i,j])**2)/(2*(self.distance_noise**2))) \
              * np.exp(-((bearing_rnl[j]-bear_meas[i,j])**2)/(2*(self.bearing_noise**2))))
             for i in range(self.num_particles) for j in range(num_landmarks)]
            ).reshape(self.num_particles, num_landmarks)
        weights = np.prod(weights/denom, axis=1)

        weight_sum = np.sum(weights)
        if weight_sum != 0.0:
            self.weights = weights / weight_sum # normalized weights
        else:
            self.weights = weights
    
    def resampling(self):
        '''
        This method performs stratified resampling of particles with their
            latest weights.

        Reference: Probabilistic Robotics, Ch.4, page 86
        '''

        new_particles = []
        new_weight = 1 / self.num_particles
        r = np.random.uniform(0, new_weight, 1)
        c = self.weights[0]
        i = 0
        count = 0

        for m in range(self.num_particles):
            u = r + (m-1)*new_weight
            while u > c:
                i += 1
                c += self.weights[i]

            x, y = self.particles[i].pos
            # Resample if this particle lies outside the map
            if abs(x) > self.map.realMapSize/2 or abs(y) > self.map.realMapSize/2:
                # Resample based on the particle with the largest weight
                idx = np.argwhere(self.weights == np.max(self.weights)).flatten()[0]
                particle = self.particles[idx]
                while np.abs(x) > self.map.realMapSize/2:
                    x = np.random.normal(particle.pos[0], 0.5, 1)
                    x = x[0]
                while np.abs(y) > self.map.realMapSize/2:
                    y = np.random.normal(particle.pos[1], 0.5, 1)
                    y = y[0]
                count += 1
                
            new_particles.append(self.Particle(x, y, self.particles[i].yaw, new_weight))

        self.particles = np.array(new_particles)
        self.weights = new_weight * np.ones(self.num_particles)

        if count != 0:
            print("Number of outliers resampled:", count)
            
    def visualize(self, image, end=False, time=None):
        """
        The function to visualize and save the result at each time step.

        Parameters:
            image (numpy array): Image of the current situation in PyBullet.
            end (bool): True = the last time step, False otherwise.
            time (int): Index of time step.
        Returns:
            None
        """

        plt.clf()
        clear_output(wait=True)

        cmap = plt.get_cmap('rainbow', self.num_particles)
        cNorm  = colors.Normalize(vmin=0, vmax=0.5)
        scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cmap)

        # Plot the map and robot
        plt.imshow(self.map.map, cmap='gray', vmin=0, vmax=1, origin='lower')
        robot_pos = self.map.World2Grid((self.robot.pos[0], self.robot.pos[1]))
        plt.scatter(robot_pos[0], robot_pos[1],
                    s=180, marker='o', color='b', edgecolors='r')
        plt.arrow(robot_pos[0], robot_pos[1],
                  18*np.cos(self.robot.yaw), 18*np.sin(self.robot.yaw),
                  head_width=3, head_length=7, length_includes_head=True,
                  color='b', linewidth=2)
        
        # Plot the particles
        max_weight = np.max(self.weights)
        max_point_idx = np.argwhere(self.weights == max_weight).flatten()[0]
        for i in range(self.num_particles+1):
            if i == self.num_particles:
                particle = self.particles[max_point_idx]
                curr_weight = max_weight
            elif i == max_point_idx:
                continue
            else:
                particle = self.particles[i]
                curr_weight = self.weights[i]

            colorVal = scalarMap.to_rgba(curr_weight)
            particle_pos = self.map.World2Grid((particle.pos[0], particle.pos[1]))
            plt.scatter(particle_pos[0], particle_pos[1],
                        marker='o', s=3, color=colorVal, cmap=scalarMap)
            plt.arrow(particle_pos[0], particle_pos[1],
                      7*np.cos(particle.yaw), 7*np.sin(particle.yaw),
                      head_width=1.5, head_length=3, length_includes_head=True,
                      color=colorVal, linewidth=0.75)
            
        plt.title('Real-time Map')
        plt.colorbar(scalarMap, label='Particle Weight', orientation='vertical', shrink=0.9)
        x, y = self.map.map.shape
        plt.xlim([-0.5, x-0.5])
        plt.ylim([0, y])

        if time is not None:
            plt.savefig('plots\\'+str(time)+'.png',
                        format='png',
                        bbox_inches ="tight",
                        orientation ='landscape')

        #### For running on Colab
        # plt.show(block=True)
        #### For running on local computer
        if end:
            plt.show(block=True)
        else:
            plt.show(block=False)
            plt.pause(0.0001)

def main():
    """
    The function to run the PF program, initialize necessary parameters, and
        visualize the simulation result.
    """

    t0 = time()
    num_of_particles = 75

    # Set up simulation speed
    sim_FPS = 0.6
    path_sim_time = Timings(sim_FPS)

    # Input a map and initialize simulation
    sim = Simulation()
    realMapSize = sim.sim.get_env_info()["map_size"]
    input_map = Map('probGridMap_perfect.txt', realMapSize, res=0.1)
    image, dataset, status, vel, steering = sim.collectData(outputImage=False, begin=True)

    pf = ParticleFilter(input_map, num_of_particles, dataset[0])

    t = 0
    time_to_plot = False
    pf.visualize(image, time=t) # display the initial particles

    # Run the PF algorithm
    while True:
        image, dataset, status, vel, steering = sim.collectData(True)

        if status == -1:
            print('Total run time:', floor((time()-t0)/60), 'min',
                  round((time()-t0)%60, 1), 'sec.')
            t += 1
            pf.visualize(image, end=True, time=t)
            break

        if path_sim_time.update_time():
            time_to_plot = True

        # Perform update once a movement is completed
        pf.prediction_step(dataset[0])
        pf.update_step()
        if np.sum(pf.weights) == 0.0:
            continue
        if time_to_plot:
            t += 1
            pf.visualize(image, time=t) # particles move based on control u
            time_to_plot = False
        pf.resampling()


if __name__ == "__main__":
    main()