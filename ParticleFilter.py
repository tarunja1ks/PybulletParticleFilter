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

# matplotlib.use('TkAgg')





class ParticleFilter:
    def __init__(self, initial_pose, OGM, numberofparticles=3):
        dataset=20
        self.numberofparticles=numberofparticles
        self.q=np.array([1.0, 0.0, 0.0, 0.0])
        self.particle_poses= np.tile(initial_pose, (self.numberofparticles, 1)).astype(np.float64)
        self.particle_weights= np.ones(self.numberofparticles)/self.numberofparticles
        
        self.quaternions=np.repeat(np.array([[1.0, 0.0, 0.0, 0.0]]),self.numberofparticles, axis=0)
        
        self.NumberEffective=numberofparticles
        self.sigma_x=0.03 # the stdev for lin vel
        self.sigma_y=0.03 # the stdev for ang vel 
        self.sigma_roll=0.0005
        self.sigma_pitch=0.0005
        self.sigma_yaw=0.005
        self.lidar_stdev=0.05
        
        self.lin_covariance=np.asarray([[self.sigma_x**2,0],[0,self.sigma_y**2]])
        self.ang_covariance=np.zeros((4, 4))
        self.ang_covariance[1,1]=self.sigma_roll**2
        self.ang_covariance[2,2]=self.sigma_pitch**2
        self.ang_covariance[3,3]=self.sigma_yaw**2
        
        
        self.xt=initial_pose

        self.prev_ang=0
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
        
    def quaternion_multiply(self, q1, q0):
        w0,x0,y0,z0= q0[:,0],q0[:,1],q0[:,2],q0[:,3]
        w1,x1,y1,z1= q1[:,0],q1[:,1],q1[:,2],q1[:,3]

        output=np.stack([
            w1*w0-x1*x0-y1*y0-z1*z0,
            w1*x0+ x1*w0+y1*z0-z1*y0,
            w1*y0-x1*z0+y1*w0+z1*x0,
            w1*z0+x1*y0-y1*x0+z1*w0
        ], axis=1)
        
        norms = np.linalg.norm(output, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  
        output /= norms

        return output
        
        
        
    def testang(self,vel,dt):
        self.q+=0.5*dt*self.quaternion_multiply(self.q,vel)
        self.q/=np.linalg.norm(self.q)
        return self.q
        
        
        
    def prediction_step(self,linU, angU, dt): # in the prediction step we create the noise and update the poses
            # lin_noise= np.random.multivariate_normal([0, 0], self.lin_covariance, size=self.numberofparticles)
            # ang_noise= np.random.multivariate_normal([0,0,0,0], self.ang_covariance, size=self.numberofparticles)
            # noisy_linU=linU+lin_noise
            # noisy_angU=angU+ang_noise
            noisy_linU=linU
            noisy_angU=angU

            # linear changes
            dx=noisy_linU[:,0]*dt/10
            dy=noisy_linU[:,1]*dt/10
            
            theta =np.linalg.norm(noisy_angU, axis=1)*dt/10 # calculating the theta 
            axis = noisy_angU / np.linalg.norm(noisy_angU, axis=1)[:, None]
            axis = np.nan_to_num(axis)
            dq=np.hstack([np.cos(theta/2)[:, None], axis * np.sin(theta/2)[:, None]])

            self.quaternions=self.quaternion_multiply(self.quaternions,dq)
            self.quaternions/=np.linalg.norm(self.quaternions, axis=1)[:, None]
            
            

            
            self.particle_poses[:,0]+=dx
            self.particle_poses[:,1]+=dy
            
            w,x,y,z=self.quaternions.T
            self.particle_poses[:,2]= np.degrees(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
            
            

            
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

    
    






    



    



