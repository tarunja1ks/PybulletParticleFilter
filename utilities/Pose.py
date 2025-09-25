import numpy as np
import math
import pybullet as p


class Pose:
    def __init__(self,x,y,h):
        self.pose=np.asarray([[math.cos(h), -math.sin(h),x],[math.sin(h),math.cos(h),y],[0,0,1]])
        self.x=x
        self.y=y
        self.h=h
        
    def getPose(self):
        return self.pose
    
    def getPoseVector(self):
        return np.asarray([self.x,self.y,self.h],dtype=float)
    
    def setPose(self,pose):
        self.pose=pose