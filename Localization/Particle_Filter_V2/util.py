import numpy as np

def World2Grid(point, realMapSize, gridMapSize, res=0.1):
    """
    A helper method to determine the grid (in grid map coordinate)
        that the given point (in world coordinate) belongs to.
        It would also be imported and used by RRT implementation.

    Parameters:
        point (tuple): World coordinate of the point.
        realMapSize (float): Side length of the real map.
        gridMapSize (int): Side length (number of grids) of the grid map.
        res (float): Grid map resolution.
    
    Returns:
        tuple: Grid map coordinate of the grid.
    """

    x, y = point[0] + realMapSize/2., point[1] + realMapSize/2.
    i, j = int(x/res), int(y/res)

    if i == gridMapSize:
        i -= 1
    if j == gridMapSize:
        j -= 1

    return (i, j)


def Grid2World(grid, realMapSize, gridMapSize, res=0.1):
    """
    A helper method to convert a given grid map coordinate
        to the world coordinate of the grid center.
        It would also be imported and used by RRT implementation.

    Parameters:
        grid (tuple): Grid map coordinate.
        realMapSize (float): Side length of the real map.
        gridMapSize (int): Side length (number of grids) of the grid map.
        res (float): Grid map resolution.
    
    Returns:
        tuple: Grid coordinate of the grid.
    """

    i, j = grid[0] * res - realMapSize/2., grid[1] * res - realMapSize/2.
    x, y = i + res/2, j + res/2

    return (x, y)

def Body2World(velocity, steering, robot_pose):
    delta_time = 1/200
    pos, yaw = robot_pose[0:2], robot_pose[2]
    T_B2W = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                      [np.sin(yaw), np.cos(yaw), 0.],
                      [0., 0., 1.]])