import gc
from OGM import OGM,Trajectory
from utils import utils as util
import matplotlib
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import time
from fractions import Fraction
from utilities.Pose import Pose
from scipy.special import erf
from tqdm import tqdm
import psutil, os
import math
import multiprocessing.resource_tracker as rt
import warnings
import torch

matplotlib.use('TkAgg')





class ParticleFilter:
    def __init__(self, initial_pose, OGM, numberofparticles=3):
        dataset=20
        self.numberofparticles=numberofparticles
        self.ang=0
        self.particle_poses= np.tile(initial_pose, (self.numberofparticles, 1)).astype(np.float64)
        self.particle_weights= np.ones(self.numberofparticles)/self.numberofparticles
        
        self.NumberEffective=numberofparticles
        self.sigma_v=0.03 # the stdev for lin vel
        self.sigma_w=0.03 # the stdev for ang vel 
        self.lidar_stdev=0.05
        
        self.covariance=np.asarray([[self.sigma_v**2,0],[0,self.sigma_w**2]])
        self.xt=initial_pose

        self.robotTosensor= np.array([OGM.sensor_x_r, OGM.sensor_y_r, OGM.sensor_yaw_r])
        self.device = torch.device('mps') # using gpu

    def normal_pdf(self,x, mu, sigma):
        return np.exp(-0.5*((x - mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))
    
    def normal_cdf(self, x, mu, sigma):
        z=(x - mu) / (sigma * np.sqrt(2))
        return 0.5 * (1 + erf(z))


        
    def getPose(self):
        return self.xt.getPose()
    def getPoseObject(self):
        return self.xt
    
    def setPose(self,pose):
        self.xt=pose
    
    def testang(self,vel,dt):
        self.ang+=vel*dt
        return self.ang
        
        
        
    def prediction_step(self,U, Tt): # in the prediction step we create the noise and update the poses
            noise= np.random.multivariate_normal([0,0], self.covariance, size=self.numberofparticles)
            noisy_U= U + noise
            vel= noisy_U[:,0]
            ang= noisy_U[:,1]
            theta= self.particle_poses[:,2]
            
            angle= ang * Tt / 2
            sinc_angle=util.sinc(angle)
            
            dx= Tt * vel * sinc_angle * np.cos(theta + angle)
            dy= Tt * vel * sinc_angle * np.sin(theta + angle)
            dtheta= Tt * ang
            
            self.particle_poses[:,0] += dx
            self.particle_poses[:,1] += dy
            self.particle_poses[:,2] += dtheta
            
    def update_step(self, OGM, scan, max_cell_range=600):

        angles=torch.tensor(np.linspace(OGM.lidar_angle_min, OGM.lidar_angle_max, len(scan)) * np.pi / 180.0,dtype=torch.float32, device=self.device)
        indValid=np.logical_and((scan < OGM.lidar_range_max), (scan > OGM.lidar_range_min))
        
        ranges=torch.tensor(scan[indValid],dtype=torch.float32, device=self.device)
        angles=angles[indValid]
    
        sensor_poses=torch.tensor(self.particle_poses,dtype=torch.float32, device=self.device) + torch.tensor(self.robotTosensor,dtype=torch.float32, device=self.device)
        sensor_x=sensor_poses[:, 0].reshape(-1, 1)
        sensor_y=sensor_poses[:, 1].reshape(-1, 1)
        sensor_angles=sensor_poses[:, 2].reshape(-1, 1)
        
        cos_sensor=torch.cos(sensor_angles)
        sin_sensor=torch.sin(sensor_angles)
        cos_angles=torch.cos(angles).reshape(1, -1)
        sin_angles=torch.sin(angles).reshape(1, -1)
        

        world_angles=sensor_angles + angles.reshape(1, -1)
        
        
        self.scales=torch.linspace(0, 1, max_cell_range, dtype=torch.float32, device=self.device).reshape(1, 1, -1)
        
        self.dx=torch.cos(world_angles)[:, :, None] * max_cell_range * self.scales
        self.dy=torch.sin(world_angles)[:, :, None] * max_cell_range * self.scales
        
        cell_sensor_x, cell_sensor_y=OGM.vector_meter_to_cell(sensor_poses.cpu().numpy().T)
        cell_sensor_x=torch.tensor(cell_sensor_x,dtype=torch.float32, device=self.device)
        cell_sensor_y=torch.tensor(cell_sensor_y,dtype=torch.float32, device=self.device)
        x_cells=torch.floor(self.dx + cell_sensor_x[:, None, None]).long()
        y_cells=torch.floor(self.dy + cell_sensor_y[:, None, None]).long()

        H, W=OGM.MAP['map'].shape
        x_cells=torch.clamp(x_cells, 0, H-1)
        y_cells=torch.clamp(y_cells, 0, W-1)
        
        
        occupied=torch.tensor(OGM.MAP['map'], dtype=torch.float32, device=self.device)[x_cells, y_cells] > 0
        
        first_occupied=torch.argmax(occupied.long(), dim=2)
        no_obstacle=~torch.any(occupied, dim=2)
        first_occupied[no_obstacle]=max_cell_range - 1
        

        particle_idx, ray_idx=torch.meshgrid(torch.arange(first_occupied.shape[0]), torch.arange(first_occupied.shape[1]), indexing='ij')
        x_hits=x_cells[particle_idx, ray_idx, first_occupied].float()
        y_hits=y_cells[particle_idx, ray_idx, first_occupied].float()
        
        ztk_star=(((y_hits-cell_sensor_y[:,None])**2+(x_hits-cell_sensor_x[:,None])**2)**0.5)/20
        
        ztk=ranges.reshape(1, -1)
        
        
        log_likelihood=-0.5 * ((ztk - ztk_star) / self.lidar_stdev)**2
        log_weights=torch.sum(log_likelihood, dim=1)

        max_log_weight=torch.max(log_weights)
        self.particle_weights=torch.exp(log_weights - max_log_weight)
        self.particle_weights /= torch.sum(self.particle_weights)
        

        weighted_x=np.sum(self.particle_poses[:, 0] * self.particle_weights.cpu().numpy().flatten())
        weighted_y=np.sum(self.particle_poses[:, 1] * self.particle_weights.cpu().numpy().flatten())
        
        weighted_sin=np.sum(np.sin(self.particle_poses[:, 2]) * self.particle_weights.cpu().numpy().flatten())
        weighted_cos=np.sum(np.cos(self.particle_poses[:, 2]) * self.particle_weights.cpu().numpy().flatten())
        weighted_angle=np.arctan2(weighted_sin, weighted_cos)
        
        self.particle_weights=self.particle_weights.cpu().numpy()
        weighted_pose=np.array([weighted_x, weighted_y, weighted_angle])
        

        
        
        return weighted_pose
    
    def resampling_step(self):
        
        self.NumberEffective= 1/np.sum(self.particle_weights**2)
        if self.NumberEffective<=self.numberofparticles * 0.3:
            cumsum= np.cumsum(self.particle_weights)
            sample_points= np.random.random() / self.numberofparticles + np.arange(self.numberofparticles) / self.numberofparticles
            indices= np.searchsorted(cumsum, sample_points)
            self.particle_poses= self.particle_poses[indices]
            self.particle_weights= np.full(self.numberofparticles, 1.0 / self.numberofparticles)

    
    






    



    



