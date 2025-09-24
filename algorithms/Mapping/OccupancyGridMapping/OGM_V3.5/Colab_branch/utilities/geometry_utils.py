import pybullet as p
import numpy as np

def Point(x=0., y=0., z=0.):
    return np.array([x, y, z])

def Euler(roll=0., pitch=0., yaw=0.):
    return np.array([roll, pitch, yaw])

def Pose(point=None, euler=None):
    point = Point() if point is None else point
    euler = Euler() if euler is None else euler
    return point, p.getQuaternionFromEuler(euler)

def Pose2d(x=0., y=0., yaw=0.):
    return np.array([x, y, yaw])
