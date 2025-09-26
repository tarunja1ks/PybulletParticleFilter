from utils import utils as util
import matplotlib
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import time
from fractions import Fraction
from utilities.Pose import Pose

import math
matplotlib.use('Qt5Agg')

class OGM: 
    def __init__(self):
        dataset=20
        # init MAP
        self.MAP= {}
        self.MAP['res']  = 0.05 #meters
        self.MAP['xmin'] = -25  #meters
        self.MAP['ymin'] = -25
        self.MAP['xmax'] =  25
        self.MAP['ymax'] =  25 
        self.MAP['sizex'] = int(np.ceil((self.MAP['xmax'] - self.MAP['xmin']) / self.MAP['res'] + 1)) #cells
        self.MAP['sizey'] = int(np.ceil((self.MAP['ymax'] - self.MAP['ymin']) / self.MAP['res'] + 1))
        self.MAP['map']= np.zeros((self.MAP['sizex'],self.MAP['sizey']),dtype=np.float32) #DATA TYPE: char or int8
        
        self.red_dot_markers = [] 
        
        fig2= plt.figure(figsize=(10, 10))
        extent= [self.MAP['ymin'], self.MAP['ymax'], self.MAP['xmin'], self.MAP['xmax']]
        self.ogm_map= plt.imshow(self.MAP['map'], cmap="gray", vmin=-5, vmax=5, 
                                 origin='lower', extent=extent)

        
        
        self.sensor_x_r= 0.30183  
        self.sensor_y_r= 0.0
        self.sensor_yaw_r= 0.0
        
        self.lidar_angle_min= -45
        self.lidar_angle_max= 45
        self.lidar_angle_increment= (135-45)/100
        self.lidar_range_min= 0.0 # minimum range value [m]
        self.lidar_range_max= 8 # maximum range value [m]
        
   
            
    def check_and_expand_map(self, x_world, y_world):
        """Check if coordinates are outside map bounds and expand if necessary"""
        expanded= False
        
        # Check if we need to expand
        x_world=x_world.ravel()
        
        minx=np.min(x_world)
        maxx=np.max(x_world)
        miny=np.min(y_world)
        maxy=np.max(y_world)
        if (minx < self.MAP['xmin'] or maxx > self.MAP['xmax'] or 
            miny < self.MAP['ymin'] or maxy > self.MAP['ymax']):
            
            # Calculate new bounds with some padding
            padding= 10  # meters
            new_xmin= min(self.MAP['xmin'], minx - padding)
            new_xmax= max(self.MAP['xmax'], maxx + padding)
            new_ymin= min(self.MAP['ymin'], miny - padding)
            new_ymax= max(self.MAP['ymax'], maxy + padding)
            
            # Calculate new map size
            new_sizex= int(np.ceil((new_xmax - new_xmin) / self.MAP['res'] + 1))
            new_sizey= int(np.ceil((new_ymax - new_ymin) / self.MAP['res'] + 1))
            
            # Create new larger map
            new_map= np.zeros((new_sizex, new_sizey), dtype=np.float32)
            
            # Calculate offset for copying old map data
            offset_x= int(np.floor((self.MAP['xmin'] - new_xmin) / self.MAP['res']))
            offset_y= int(np.floor((self.MAP['ymin'] - new_ymin) / self.MAP['res']))
            
            # Copy old map data to new map
            new_map[offset_x:offset_x + self.MAP['sizex'], 
                   offset_y:offset_y + self.MAP['sizey']]= self.MAP['map']
            
            # Update map parameters
            self.MAP['xmin']= new_xmin
            self.MAP['xmax']= new_xmax
            self.MAP['ymin']= new_ymin
            self.MAP['ymax']= new_ymax
            self.MAP['sizex']= new_sizex
            self.MAP['sizey']= new_sizey
            self.MAP['map']= new_map
            expanded= True
            print("just expanded")
            
        return expanded
            
    def meter_to_cell(self, pose_vector):
        
        x= pose_vector[0]
        
        y= pose_vector[1]
        
        # Check and expand map if necessary
        self.check_and_expand_map(x, y)
        
        # Convert to cell coordinates using proper floor operation
        cell_x= int(np.floor((x - self.MAP['xmin']) / self.MAP['res']))
        cell_y= int(np.floor((y - self.MAP['ymin']) / self.MAP['res']))
        
        # Clamp to valid range (should rarely be needed now)
        cell_x= max(0, min(cell_x, self.MAP['sizex'] - 1))
        cell_y= max(0, min(cell_y, self.MAP['sizey'] - 1))
        
        return cell_x, cell_y
    
    def vector_meter_to_cell(self, pose_vector):
        
        x= pose_vector[0]
        
        y= pose_vector[1]
        
        
        # Check and expand map if necessary
        self.check_and_expand_map(x, y)

        # Convert to cell coordinates using proper floor operation
        cell_x= np.array(np.floor((x - self.MAP['xmin']) / self.MAP['res']), dtype=int)
        cell_y= np.array(np.floor((y - self.MAP['ymin']) / self.MAP['res']), dtype=int)
        
        # Clamp to valid range (should rarely be needed now)
        cell_x=np.clip(cell_x,0,self.MAP['sizex'] - 1)
        cell_y=np.clip(cell_y, 0,self.MAP['sizey'] - 1)
        
       
        return cell_x, cell_y
    
    
    
    def plot_red_dot(self, cell_x, cell_y):
        # Convert cell coordinates back to world coordinates
        world_x = self.MAP['xmin'] + cell_x * self.MAP['res']
        world_y = self.MAP['ymin'] + cell_y * self.MAP['res']
        
        
        # Get the axes that the ogm_map belongs to
        ax = self.ogm_map.axes
        
        # Note: your plot has x,y swapped in the extent, so plot as (y,x)
        marker, = ax.plot(world_y, world_x, 'ro', markersize=1, zorder=10)
        self.red_dot_markers.append(marker)
        
        # Force immediate update on the same figure as the map
        ax.figure.canvas.draw()
        ax.figure.canvas.flush_events()
        # plt.pause(0.01)
            
        
    def ogm_plot(self, x, y, occupied=False, scale=1, bound=10):
        if not (0 <= x < self.MAP['sizex'] and 0 <= y < self.MAP['sizey']):
            return
        confidence= 0.9 # confidence level of the sensor
        if occupied:
            odds= confidence / (1 - confidence)
        else:
            odds= (1 - confidence) / confidence
        self.MAP['map'][x][y] += (math.log(odds))*scale
        self.MAP['map'][x][y]= max(-bound, min(bound, self.MAP['map'][x][y]))
    
    def ogm_plot_vectorized(self, x, y, occupied=False, scale=1, bound=10):
        if (x.min() < 0 or x.max() >= self.MAP['sizex'] or y.min() < 0 or y.max() >= self.MAP['sizey']):
            return
        
        
        confidence= 0.9 # confidence level of the sensor
        if occupied:
            odds= confidence / (1 - confidence)
        else:
            odds= (1 - confidence) / confidence
        
        self.MAP['map'][x,y]= np.clip(self.MAP['map'][x,y]+(math.log(odds))*scale ,-bound,bound) # clipping each thing between bounds after adding log odds
        
        
    def logOddstoProbability(self,logOdds):
        return 1 / (1 + math.exp(-logOdds))
    def probabilityToLogOdds(self,probability):
        return math.log(probability/(1-probability))
    
    def bressenham_mark_Cells(self, scan, robot_pose):
        
        
        angles=np.arange(self.lidar_angle_min, self.lidar_angle_max,self.lidar_angle_increment)*np.pi/180

        valid=(scan>=self.lidar_range_min)&(scan<=self.lidar_range_max)
        ranges=scan[valid]
        angles=angles[valid]
        xs=ranges*np.cos(angles)
        ys=ranges*np.sin(angles)
        
        robot_pose[2]=robot_pose[2]*np.pi/180
        
        c,s=np.cos(robot_pose[2]),np.sin(robot_pose[2])
        
        rot=np.array([[c,-s],[s,c]])
        
        print(ranges)
        world = np.dot(rot, np.vstack([xs,ys])) + (robot_pose[:2])[:,None]

        sensor_cell=self.meter_to_cell(robot_pose[:2])
        x_cell,y_cell=self.vector_meter_to_cell(world)  # this is convering all the hit points into grid coordinates instead of real-world
        
        sx=np.repeat(sensor_cell[0], len(ranges))
        sy=np.repeat(sensor_cell[1],len(ranges))
        
        
        
        line_x,line_y=util.bresenham2D_vectorized(sx,sy,x_cell,y_cell)

        
        self.ogm_plot_vectorized(np.array(line_x,dtype=int),np.array(line_y,dtype=int),False)
        

        validOcc=(scan>=self.lidar_range_min)&(scan<self.lidar_range_max)
        num_valid = np.count_nonzero(validOcc)   
        if num_valid == 0:
            return
        
        Occupied=scan[validOcc]
        angles2=np.arange(self.lidar_angle_min, self.lidar_angle_max,self.lidar_angle_increment)*np.pi/180
        angles2=angles2[validOcc]
        xs2=Occupied*np.cos(angles2)
        ys2=Occupied*np.sin(angles2)
        
        world2 = np.dot(rot, np.vstack([xs2,ys2])) + (robot_pose[:2])[:,None]
        x_cell2,y_cell2=self.vector_meter_to_cell(world2)
        
        # self.ogm_plot_vectorized(x_cell2,y_cell2,True)
    
    

    def showPlots(self):
        plt.show()
    
    # def mapCorrelation(): # making it again to understand it more 
        
    def updatePlot(self, robot_pose=None):
        current_extent = [self.MAP['ymin'], self.MAP['ymax'], self.MAP['xmin'], self.MAP['xmax']]
        
        # Update data directly
        self.ogm_map.set_data(self.MAP['map'])
        self.ogm_map.set_extent(current_extent)
        
        if robot_pose is not None:
            pose_vec = robot_pose.getPoseVector()
            self.robot_marker.set_data([pose_vec[1]], [pose_vec[0]])
        
        
        plt.pause(0.1)
        print("updddddddate plot")
        

    
        
        
class Trajectory:
    def __init__(self, initial_pose_vector):
        self.fig_traj, self.ax_traj= plt.subplots(1, 1, figsize=(8, 8))
        self.ax_traj.set_title("Robot Trajectory")
        self.ax_traj.set_xlabel("X [m]")
        self.ax_traj.set_ylabel("Y [m]")
        self.ax_traj.set_aspect('equal', adjustable='box') 
        self.ax_traj.grid(True)
        
        # initializing robot position
        self.trajectory_x= []
        self.trajectory_y= []
        self.trajectory_h=[]
        self.trajectory_x.append(initial_pose_vector[0])
        self.trajectory_y.append(initial_pose_vector[1])
        self.trajectory_h.append(initial_pose_vector[2])
        
        self.trajectory_line_traj,= self.ax_traj.plot(self.trajectory_x, self.trajectory_y, 'b-', linewidth=2, label='Trajectory')
        
    def showPlot(self):
        self.trajectory_line_traj.set_data(self.trajectory_x, self.trajectory_y)
        self.ax_traj.relim() 
        self.ax_traj.autoscale_view() 
        self.fig_traj.canvas.draw_idle()
        self.fig_traj.canvas.flush_events()