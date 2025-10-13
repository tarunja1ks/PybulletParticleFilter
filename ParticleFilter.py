"""
Particle Filter implementation for robot localization using LIDAR data.

This module provides a Monte Carlo localization algorithm that estimates
robot pose in a known environment using particle filtering techniques.
"""

# Standard library imports
from fractions import Fraction

# Third-party imports
import matplotlib.pyplot as plt; plt.ion()
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import erf
from tqdm import tqdm

# Local imports
from OGM import OGM, Trajectory

# Configure matplotlib for interactive plotting
plt.ion()


class ParticleFilter:
    """
    A Monte Carlo localization particle filter for robot pose estimation.
    
    This class implements a particle filter algorithm that uses LIDAR
    measurements to estimate the pose of a robot in a known environment
    represented by an Occupancy Grid Map (OGM).
    
    Attributes:
        number_of_particles (int): Number of particles in the filter
        particle_poses (np.ndarray): Array of particle poses [x, y, theta]
        particle_weights (np.ndarray): Normalized weights for each particle
        quaternions (np.ndarray): Quaternion representations for particles
        number_effective (float): Effective number of particles
        device (torch.device): Computing device (CPU/CUDA/MPS)
    """
    
    def __init__(self, initial_pose, ogm, number_of_particles=3):
        """
        Initialize the particle filter.
        
        Args:
            initial_pose (np.ndarray): Initial robot pose [x, y, theta]
            ogm (OGM): Occupancy Grid Map object
            number_of_particles (int): Number of particles to use
        """
        self.number_of_particles = number_of_particles
        self.particle_poses = np.tile(
            initial_pose, 
            (self.number_of_particles, 1)
        ).astype(np.float64)
        self.particle_weights = np.ones(self.number_of_particles) / self.number_of_particles
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.xt = initial_pose
        
        self.quaternions = np.repeat(
            np.array([[1.0, 0.0, 0.0, 0.0]]),
            self.number_of_particles, 
            axis=0
        )
        
        self.number_effective = number_of_particles
        
        # Initialize noise parameters
        self._init_stddevs(
            sigma_x=0.03,
            sigma_y=0.03,
            sigma_roll=0.000,
            sigma_pitch=0.000,
            sigma_yaw=0.000001,
            lidar_stdev=0.01
        )
        self._init_covariances()
        
        self.prev_ang = 0
        self.robot_to_sensor = np.array([
            ogm.sensor_x_r,
            ogm.sensor_y_r,
            ogm.sensor_yaw_r
        ])
        
        # Set up computing device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else
            'cpu'
        )
    
    def _init_stddevs(self, sigma_x, sigma_y, sigma_roll, sigma_pitch, sigma_yaw, lidar_stdev):
        """
        Initialize standard deviations for motion and sensor models.
        
        Args:
            sigma_x (float): Standard deviation for x-direction motion
            sigma_y (float): Standard deviation for y-direction motion
            sigma_roll (float): Standard deviation for roll rotation
            sigma_pitch (float): Standard deviation for pitch rotation
            sigma_yaw (float): Standard deviation for yaw rotation
            lidar_stdev (float): Standard deviation for LIDAR measurements
        """
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_roll = sigma_roll
        self.sigma_pitch = sigma_pitch
        self.sigma_yaw = sigma_yaw
        self.lidar_stdev = lidar_stdev
    
    def _init_covariances(self):
        """Initialize covariance matrices for motion and sensor models."""
        self.lin_covariance = np.asarray([
            [self.sigma_x**2, 0],
            [0, self.sigma_y**2]
        ])
        
        self.ang_covariance = np.zeros((3, 3))
        self.ang_covariance[0, 0] = self.sigma_roll**2
        self.ang_covariance[1, 1] = self.sigma_pitch**2
        self.ang_covariance[2, 2] = self.sigma_yaw**2
    
    def normal_pdf(self, x, mu, sigma):
        """
        Calculate the probability density function of a normal distribution.
        
        Args:
            x (float): Value to evaluate
            mu (float): Mean of the distribution
            sigma (float): Standard deviation of the distribution
            
        Returns:
            float: PDF value at x
        """
        return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    
    def normal_cdf(self, x, mu, sigma):
        """
        Calculate the cumulative distribution function of a normal distribution.
        
        Args:
            x (float): Value to evaluate
            mu (float): Mean of the distribution
            sigma (float): Standard deviation of the distribution
            
        Returns:
            float: CDF value at x
        """
        z = (x - mu) / (sigma * np.sqrt(2))
        return 0.5 * (1 + erf(z))
    
    def get_pose(self):
        """
        Get the current estimated pose.
        
        Returns:
            The current pose estimate
        """
        return self.xt.getPose()
    
    def get_pose_object(self):
        """
        Get the current pose object.
        
        Returns:
            The current pose object
        """
        return self.xt
    
    def set_pose(self, pose):
        """
        Set the current pose.
        
        Args:
            pose: New pose to set
        """
        self.xt = pose
    
    def quaternion_multiply(self, q1, q0):
        """
        Multiply two quaternions.
        
        Args:
            q1 (np.ndarray): First quaternion array [w, x, y, z]
            q0 (np.ndarray): Second quaternion array [w, x, y, z]
            
        Returns:
            np.ndarray: Product of quaternions, normalized
        """
        w0, x0, y0, z0 = q0[:, 0], q0[:, 1], q0[:, 2], q0[:, 3]
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        
        output = np.stack([
            w1*w0 - x1*x0 - y1*y0 - z1*z0,
            w1*x0 + x1*w0 + y1*z0 - z1*y0,
            w1*y0 - x1*z0 + y1*w0 + z1*x0,
            w1*z0 + x1*y0 - y1*x0 + z1*w0
        ], axis=1)
        
        norms = np.linalg.norm(output, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        output /= norms
        
        return output
    
    def test_ang(self, vel, dt):
        """
        Test angular velocity integration using quaternions.
        
        Args:
            vel: Angular velocity
            dt (float): Time step
            
        Returns:
            np.ndarray: Updated quaternion
        """
        self.q += 0.5 * dt * self.quaternion_multiply(self.q, vel)
        self.q /= np.linalg.norm(self.q)
        return self.q
    
    def prediction_step(self, lin_u, ang_u, dt):
        """
        Perform the prediction step of the particle filter.
        
        This step propagates particles forward in time based on control inputs
        and motion model noise.
        
        Args:
            lin_u (np.ndarray): Linear velocity control input [vx, vy]
            ang_u (np.ndarray): Angular velocity control input [wx, wy, wz]
            dt (float): Time step
        """
        # Generate noise for motion model
        lin_noise = np.random.multivariate_normal(
            [0, 0],
            self.lin_covariance,
            size=self.number_of_particles
        )
        ang_noise = np.random.multivariate_normal(
            [0, 0, 0],
            self.ang_covariance,
            size=self.number_of_particles
        )
        
        # Add noise to control inputs
        noisy_lin_u = lin_u + lin_noise
        noisy_ang_u = ang_u + ang_noise
        
        # Calculate linear displacement
        dx = noisy_lin_u[:, 0] * dt
        dy = noisy_lin_u[:, 1] * dt
        
        # Calculate angular displacement using quaternions
        theta = np.linalg.norm(noisy_ang_u, axis=1) * dt
        axis = noisy_ang_u / np.linalg.norm(noisy_ang_u, axis=1)[:, None]
        axis = np.nan_to_num(axis)
        
        xyz = axis * np.sin(theta / 2)[:, None]
        w = np.cos(theta / 2)[:, None]
        
        dq = np.hstack([w, xyz])
        
        # Update quaternions and poses
        self.quaternions = self.quaternion_multiply(dq, self.quaternions)
        self.quaternions /= np.linalg.norm(self.quaternions, axis=1)[:, None]
        
        self.particle_poses[:, 0] += dx
        self.particle_poses[:, 1] += dy
        
        # Convert quaternions back to Euler angles
        w, x, y, z = self.quaternions.T
        self.particle_poses[:, 2] = np.arctan2(
            2 * (w*z + x*y),
            1 - 2 * (y*y + z*z)
        )
    
    def update_step(self, ogm, scan, max_cell_range=600):
        """
        Perform the update step of the particle filter.
        
        This step updates particle weights based on LIDAR measurements
        and the occupancy grid map.
        
        Args:
            ogm (OGM): Occupancy Grid Map object
            scan (np.ndarray): LIDAR scan data
            max_cell_range (int): Maximum range in grid cells for ray tracing
            
        Returns:
            np.ndarray: Weighted pose estimate [x, y, theta]
        """
        # Convert LIDAR angles to radians
        angles = torch.tensor(
            np.linspace(
                ogm.lidar_angle_min,
                ogm.lidar_angle_max,
                len(scan)
            ) * np.pi / 180.0,
            dtype=torch.float32,
            device=self.device
        )
        
        # Filter valid measurements
        ind_valid = np.logical_and(
            (scan < ogm.lidar_range_max),
            (scan > ogm.lidar_range_min)
        )
        
        ranges = torch.tensor(
            scan[ind_valid],
            dtype=torch.float32,
            device=self.device
        )
        angles = angles[ind_valid]
        
        # Calculate sensor poses
        sensor_poses = (
            torch.tensor(self.particle_poses, dtype=torch.float32, device=self.device) +
            torch.tensor(self.robot_to_sensor, dtype=torch.float32, device=self.device)
        )
        sensor_angles = sensor_poses[:, 2].reshape(-1, 1)
        world_angles = sensor_angles + angles.reshape(1, -1)
        
        # Ray tracing setup
        self.scales = torch.linspace(
            0, 1, max_cell_range,
            dtype=torch.float32,
            device=self.device
        ).reshape(1, 1, -1)
        
        self.dx = torch.cos(world_angles)[:, :, None] * max_cell_range * self.scales
        self.dy = torch.sin(world_angles)[:, :, None] * max_cell_range * self.scales
        
        # Convert sensor positions to grid coordinates
        cell_sensor_x, cell_sensor_y = ogm.vector_meter_to_cell(
            sensor_poses.cpu().numpy().T
        )
        cell_sensor_x = torch.tensor(
            cell_sensor_x,
            dtype=torch.float32,
            device=self.device
        )
        cell_sensor_y = torch.tensor(
            cell_sensor_y,
            dtype=torch.float32,
            device=self.device
        )
        
        # Calculate ray end points
        x_cells = torch.floor(self.dx + cell_sensor_x[:, None, None]).long()
        y_cells = torch.floor(self.dy + cell_sensor_y[:, None, None]).long()
        
        # Clamp to map boundaries
        H, W = ogm.MAP['map'].shape
        x_cells = torch.clamp(x_cells, 0, H - 1)
        y_cells = torch.clamp(y_cells, 0, W - 1)
        
        # Find first occupied cell along each ray
        occupied = torch.tensor(
            ogm.MAP['map'],
            dtype=torch.float32,
            device=self.device
        )[x_cells, y_cells] > 0
        
        first_occupied = torch.argmax(occupied.long(), dim=2)
        no_obstacle = ~torch.any(occupied, dim=2)
        first_occupied[no_obstacle] = max_cell_range - 1
        
        # Calculate expected ranges
        particle_idx, ray_idx = torch.meshgrid(
            torch.arange(first_occupied.shape[0]),
            torch.arange(first_occupied.shape[1]),
            indexing='ij'
        )
        x_hits = x_cells[particle_idx, ray_idx, first_occupied].float()
        y_hits = y_cells[particle_idx, ray_idx, first_occupied].float()
        
        ztk_star = (
            ((y_hits - cell_sensor_y[:, None]) ** 2 +
             (x_hits - cell_sensor_x[:, None]) ** 2) ** 0.5
        ) / 20
        
        # Calculate likelihood and update weights
        ztk = ranges.reshape(1, -1)
        log_likelihood = -0.5 * ((ztk - ztk_star) / self.lidar_stdev) ** 2
        log_weights = torch.sum(log_likelihood, dim=1)
        
        max_log_weight = torch.max(log_weights)
        self.particle_weights = torch.exp(log_weights - max_log_weight)
        self.particle_weights /= torch.sum(self.particle_weights)
        
        # Calculate weighted pose estimate
        weights_np = self.particle_weights.cpu().numpy().flatten()
        
        weighted_x = np.sum(self.particle_poses[:, 0] * weights_np)
        weighted_y = np.sum(self.particle_poses[:, 1] * weights_np)
        
        weighted_sin = np.sum(np.sin(self.particle_poses[:, 2]) * weights_np)
        weighted_cos = np.sum(np.cos(self.particle_poses[:, 2]) * weights_np)
        weighted_angle = np.arctan2(weighted_sin, weighted_cos)
        
        self.particle_weights = weights_np
        
        weighted_pose = np.array([weighted_x, weighted_y, weighted_angle])
        weighted_pose[2] = np.degrees(weighted_pose[2])
        weighted_pose[1] = -weighted_pose[1]
        
        return weighted_pose
    
    def resampling_step(self):
        """
        Perform resampling of particles based on their weights.
        
        Uses systematic resampling when the effective number of particles
        falls below 30% of the total number of particles.
        """
        self.number_effective = 1 / np.sum(self.particle_weights**2)
        
        if self.number_effective <= (self.number_of_particles * 0.3):
            cumsum = np.cumsum(self.particle_weights)
            sample_points = (
                np.random.random() / self.number_of_particles +
                np.arange(self.number_of_particles) / self.number_of_particles
            )
            indices = np.searchsorted(cumsum, sample_points)
            
            self.particle_poses = self.particle_poses[indices]
            self.particle_weights = np.full(
                self.number_of_particles,
                1.0 / self.number_of_particles
            )
