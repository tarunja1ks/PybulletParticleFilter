# ------------------------------------------------------------------
# Implementation of Rapidly-Exploring Random Tree
# (RRT) | Status: COMPLETED (Version 1.2: Unbiased sampling)
#       | Contributor: Jeffrey Chen
#
# Update from previous version:
# - Generic world map size
#
# Key assumptions:
# - Raw world map are known.
# - RRT is probabilistically complete, but not optimal.
# - Square grid map
#
# Function:
# - Given map, start point, and goal, generate a tree graph and
#   a path from start to goal node.

# Reference:
# - Prof. Atanasov, UCSD ECE276B Lecture: Sampling Based Planning
# ------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from algorithms.Mapping.OccupancyGridMapping.Occupancy_Grid_Mapping import World2Grid, Grid2World
from math import *

class RRT():
    """
    A class used to implement RRT algorithm and provide visualization.

    Attributes:
        start (tuple): World coordinate of start point.
        goal (tuple): World coordinate of goal point.
        obstacleList (list): List of obstacles (grids) in grid coordinate.
        V_nodeList (list): List of added nodes during RRT operation.
        epsilon (float): Step size in world-frame unit.
        robotRadius (float): Size of robot in world-frame unit.
    """


    class Node():
        """
        A sub-class used to store information about each node.

        Attributes:
            x (float): X coordinate of Node in world-frame unit.
            y (float): Y coordinate of Node in world-frame unit.
            parent (Node): The parental node of this object.
        """
        
        def __init__(self, x, y):
            """
            Constructor of Node to create a new Node object.

            Parameters:
                x (float): X coordinate of Node in world-frame unit.
                y (float): Y coordinate of Node in world-frame unit.
            """

            self.x = x
            self.y = y
            self.parent = None


    class GridMap:
        """
        A sub-class used to store information about the grid map.

        Attributes:
            worldMapSize_x (float): Side length of world map in x-axis.
            worldMapSize_y (float): Side length of world map in y-axis.
            gridMap (2D numpy array): Array of occupancy probabilities of all grids.
            gridMapSize (int): Side length of grid map (number of grids).
            resolution (float): Resolution of grid map (side length of each grid).
        """

        def __init__(self,
                     gridMapFile,
                     worldMapSize_x,
                     worldMapSize_y,
                     gridMapSize,
                     resolution,
                     ):
            """
            Constructor of GridMap to create a new GridMap object.

            Parameters:
                gridMapFile (str): File path of grid map file (.txt).
                worldMapSize_x (float): Side length of world map in x-axis.
                worldMapSize_y (float): Side length of world map in y-axis.
                gridMapSize (int): Side length of grid map (number of grids).
                resolution (float): Resolution of grid map (side length of each grid).
            """
            
            self.worldMapSize_x = worldMapSize_x
            self.worldMapSize_y = worldMapSize_y
            self.gridMap = np.loadtxt(gridMapFile)
            self.gridMapSize = gridMapSize
            self.resolution = resolution


    def __init__(self, startPoint, goalPoint, epsilon, robotRadius):
        """
        Constructor of RRT to initialize a new RRT operation.

        Parameters:
            startPoint (tuple): World coordinate of start point.
            goalPoint (tuple): World coordinate of goal point.
            epsilon (float): Step size in world-frame unit.
            robotRadius (float): Size of robot in world-frame unit.
        """

        self.start = self.Node(startPoint[0], startPoint[1])
        self.goal = self.Node(goalPoint[0], goalPoint[1])
        self.obstacleList = None
        self.V_nodeList = [self.start]
        self.epsilon = epsilon
        self.robotRadius = robotRadius


    def getPath(self, gridMapObj, epsilon, obstacleList, robotRadius):
        """
        The function to perform node generation and path planning.

        Parameters:
            gridMapObj (GridMap): Object storing information about the grid map.
            epsilon (float): Step size in world-frame unit.
            obstacleList (list): List of obstacles (grids) in grid coordinate.
            robotRadius (float): Size of robot in world-frame unit.

        Returns:
            RRTpath (list): List of Node that form the path.
        """

        count = 0 # Iteration count
        maxIter = 500 # Maximum iteration

        # Keep exploring until goal is linked to a node
        while self.goal.parent == None:
            # Get a randomly sampled point
            samplePoint = self.getRandomPoint(gridMapObj.worldMapSize_x,
                                              gridMapObj.worldMapSize_y)
            
            nearestNode = self.getNearestNode(self.V_nodeList, samplePoint)

            # Find the distance b/w nearestNode and goal to determine if okay to exit
            dist2Goal = sqrt((nearestNode.x-self.goal.x)**2 + (nearestNode.y-self.goal.y)**2)
            if dist2Goal <= epsilon:
                self.goal.parent = nearestNode
                break

            # Obtain the new node and add it
            newNode = self.steer(nearestNode, samplePoint, epsilon,
                                 obstacleList, gridMapObj, robotRadius)

            if newNode != None:
                self.V_nodeList.append(newNode)
                self.animate(newNode, gridMapObj, samplePoint)

            # Exit if reaching maximum iteration
            count += 1
            if count == maxIter:
                print("Mission Failed: reached max iteration limit!")
                return [self.start, self.goal]
            
        # Generate the path from start to goal
        RRTpath, curNode = [self.goal], self.goal

        while curNode.parent != None:
            RRTpath.insert(0, curNode.parent)
            curNode = curNode.parent

        return RRTpath

    
    def getObstacleList(self, probGridMap):
        """
        The function to obtain the obstacle list based on the given grid map.
            In this case, the undetermined areas existing in the given grid map
            are considered occupied.
        
        Note: 0 = occupied, 1 = free (for visualization purpose).

        Parameters:
            probGridMap (2D numpy array): Array of occupancy probabilities of all grids.

        Returns:
            obstacleList (list): List of obstacles (grids) in grid coordinate.
        """

        obstacleList = []

        # Iterate thru each grid to find out all occupied ones.
        for row in range(probGridMap.shape[0]):
            for col in range(probGridMap.shape[1]):
                # Detect if the grid is occupied (assume undertermined = occupied)
                if (probGridMap[row][col] != 1):
                    obstacleList.append((row, col))

        return obstacleList
    

    def getRandomPoint(self, mapsize_x, mapsize_y):
        """
        The function to generate random points in the range of world map size.

        Note: 1/ Assume the origin is at the center of the world map.
              2/ Goal-unbiased sampling.

        Parameters:
            mapsize_x (float): Side length of world map in x-axis.
            mapsize_y (float): Side length of world map in y-axis.

        Returns:
            tuple: A random point in world-frame coordinate.
        """

        sample_x = np.random.uniform(-mapsize_x/2, mapsize_x/2)
        sample_y = np.random.uniform(-mapsize_y/2, mapsize_y/2)
        return (sample_x, sample_y) # assume origin at world map center
    
    
    def getRelativePosition(self, node, point):
        """
        The function to calculate a point's relative position in world frame
            (position and orientation) based on the local frame of a node.

        Parameters:
            node (Node): A Node in the map.
            point (tuple): A point whose relative position is desired.

        Returns:
            distance (float): L2 norm between the two points.
            orientation (rad): Relative angle between the two points.
        """

        # Use L2 norm as distance
        dx, dy = point[0] - node.x, point[1] - node.y
        distance = sqrt(dx**2 + dy**2)
        orientation = atan2(dy, dx)
        return distance, orientation
    

    def getNearestNode(self, V_nodeList, target):
        """
        The function to obtain the node that is closest to the target.

        Parameters:
            V_nodeList (float): List of added Node.
            target (tuple): A point whose nearest Node is desired.

        Returns:
            nearestNode (Node): The Node closest to target.
        """

        minDist = float('Inf')
        nearestNode = None

        for node in V_nodeList:
            # Use L2 norm as distance
            dist, ang = self.getRelativePosition(node, target)
            
            if dist < minDist:
                minDist = dist
                nearestNode = node

        return nearestNode


    def steer(self, nearestNode, randomPoint, epsilon, obstacleList, gridMapObj, robotRadius):
        """
        The function to obtain a new Node and return None if it's in collision.

        Parameters:
            nearestNode (Node): The Node closest to target.
            randomPoint (tuple): The randomly sampled point that determines
                the exploring direction.
            epsilon (float): Step size in world-frame unit.
            obstacleList (list): List of obstacles (grids) in grid coordinate.
            gridMapObj (GridMap): Object storing information about the grid map.
            robotRadius (float): Size of robot in world-frame unit.

        Returns:
            newNode (Node): The new Node that will be added to node list.
        """

        dist, ang = self.getRelativePosition(nearestNode, randomPoint)

        # Steer from nearest node towards random point by a step size
        newPoint = (epsilon*cos(ang) + nearestNode.x,
                    epsilon*sin(ang) + nearestNode.y)
        
        # Check if the new point collides with any obstacle
        if not self.isCollide(nearestNode, newPoint, obstacleList, gridMapObj,
                              robotRadius):
            
            # Generate and link the new node to the nearest node
            newNode = self.Node(newPoint[0], newPoint[1])
            newNode.parent = nearestNode
            return newNode

        return None
    

    def Point2Grid(self, point, gridMapSize, resolution):
        """
        A helper method to determine the grid (in grid map coordinate)
            that the given point (in world coordinate) belongs to.

        Parameters:
            point (tuple): World coordinate of the point.
            gridMapSize (int): Side length of the grid map.
            resolution (float): Grid map resolution.

        Returns:
            tuple: Grid map coordinate of the grid.
        """

        return World2Grid(point, gridMapSize, resolution)
    

    def getGridCenter(self, grid, gridMapSize, resolution):
        """
        A helper method to obtain the grid center in world coordinate.

        Parameters:
            grid (tuple): Grid map coordinate.
            gridMapSize (int): Side length of the grid map.
            resolution (float): Grid map resolution.
    
        Returns:
            tuple: World coordinate of the grid center.
        """
        
        return Grid2World(grid, gridMapSize, resolution)
    

    def isCollide(self, startNode, endPoint, obstacleList, gridMapObj, robotRadius):
        """
        The function to check if a segment is in collision with obstacles.

        Parameters:
            startNode (Node): The Node as one of the ends of the segment.
            endPoint (tuple): The point as the other end of the segment.
            obstacleList (list): List of obstacles (grids) in grid coordinate.
            gridMapObj (GridMap): Object storing information about the grid map.
            robotRadius (float): Size of robot in world-frame unit.
            
        Returns:
            boolean: True if in collision, free otherwise.
        """

        gridMapSize, res = gridMapObj.gridMapSize, gridMapObj.resolution

        # Obtain the grid centers of all obstacles
        gridCenters = [self.getGridCenter(grid, gridMapSize, res)
                       for grid in obstacleList]

        # Iter thru each point of the segment (separated by a fixed dist of res)
        for alpha in np.arange(0, 1, res):

            # Get the point's coordinates
            point_x = alpha*startNode.x + (1-alpha)*endPoint[0]
            point_y = alpha*startNode.y + (1-alpha)*endPoint[1]

            # Determine the grid that this point belongs to
            pointGrid = self.Point2Grid((point_x, point_y), gridMapSize, res)

            if pointGrid in obstacleList:
                return True # collision
            
            # Find out the gap between this point and the closest grid
            distList = [sqrt((point_x-grid[0])**2 + (point_y-grid[1])**2)
                        for grid in gridCenters]
            minDist = min(distList) - sqrt(2)*res/2

            if minDist <= robotRadius:
                return True # collision
            
        return False # collision-free
    

    def animate(self, newNode, gridMapObj, samplePoint):
        """
        The function to do the animation to see how the graph develops over time.
            It scales and displays all nodes and tree graph in the grid frame.

        Parameters:
            newNode (Node): The Node that is newly added to the graph.
            gridMapObj (GridMap): Object storing information about the grid map.
            samplePoint (tuple): The randomly sampled point within the world map.
            
        Returns:
            None
        """

        gridMapSize, res = gridMapObj.gridMapSize, gridMapObj.resolution

        # Clear the existing figure
        plt.clf()

        # Plot the grid map
        plt.imshow(gridMapObj.gridMap, cmap='gray', vmin = 0, vmax = 1, origin='lower')

        # Plot the existing tree graph (Green edges)
        for node in self.V_nodeList:
            if node.parent != None:
                nodePoint = self.Point2Grid((node.x, node.y), gridMapSize, res)
                parentPoint = self.Point2Grid((node.parent.x, node.parent.y), gridMapSize, res)
                plt.plot([nodePoint[0], parentPoint[0]],
                         [nodePoint[1], parentPoint[1]],
                         '-g')

        # Plot start node (Blue diamond)
        start = self.Point2Grid((self.start.x, self.start.y), gridMapSize, res)
        plt.plot(start[0], start[1], marker='D', markersize=5, c='b')
        plt.text(start[0], start[1], 'start', va="top", ha="right", c='b')

        # Plot goal node (Blue diamond)
        goal = self.Point2Grid((self.goal.x, self.goal.y), gridMapSize, res)
        plt.plot(goal[0], goal[1], marker='D', markersize=5, c='b')
        plt.text(goal[0], goal[1]+0.8, 'goal', va="bottom", ha="left", c='b')

        # Plot sampled point (Magenta star)
        sample = self.Point2Grid(samplePoint, gridMapSize, res)
        plt.plot(sample[0], sample[1], '*m', markersize=6, label='Sample Point')

        # Plot the new node (Red filled '+')
        newPoint = self.Point2Grid((newNode.x, newNode.y), gridMapSize, res)
        plt.plot(newPoint[0], newPoint[1], 'Pr', label='New Node')

        plt.legend(loc='upper left')

        plt.pause(0.05)


    def plotPath(self, gridMapObj, path):
        """
        The function to plot the final result of this RRT operation.

        Parameters:
            gridMapObj (GridMap): Object storing information about the grid map.
            path (list): List of Node that form the path.
            
        Returns:
            None
        """

        gridMapSize, res = gridMapObj.gridMapSize, gridMapObj.resolution

        # Clear the existing figure
        plt.clf()

        # Plot the grid map
        plt.imshow(gridMapObj.gridMap, cmap='gray', vmin = 0, vmax = 1, origin='lower')

        # Plot the existing tree graph (Green edges)
        for node in self.V_nodeList:
            if node.parent != None:
                nodePoint = self.Point2Grid((node.x, node.y), gridMapSize, res)
                parentPoint = self.Point2Grid((node.parent.x, node.parent.y), gridMapSize, res)
                plt.plot([nodePoint[0], parentPoint[0]],
                         [nodePoint[1], parentPoint[1]],
                         '-g')

        # Convert the path list to grid coords for visualization purpose
        path_x_list, path_y_list = [], []
        for node in path:
            grid = self.Point2Grid((node.x, node.y), gridMapSize, res)
            path_x_list.append(grid[0])
            path_y_list.append(grid[1])

        # Plot start node (Blue diamond)
        plt.plot(path_x_list[0], path_y_list[0], marker='D', markersize=5, c='b')
        plt.text(path_x_list[0], path_y_list[0]-0.8, 'start', va="top", ha="right", c='b', fontsize=11)

        # Plot goal node (Blue diamond)
        plt.plot(path_x_list[-1], path_y_list[-1], marker='D', markersize=5, c='b')
        plt.text(path_x_list[-1], path_y_list[-1]+0.8, 'goal', va="bottom", ha="left", c='b', fontsize=11)

        # Check if the path list contains start and goal nodes only
        if len(path) == 2:
            plt.title("Mission Failed: No path found before reaching iteration limit!", c='r')
            plt.show()
            return

        # Plot the final path
        plt.plot(path_x_list, path_y_list, '-or', markersize=4)
        plt.plot(path_x_list[0], path_y_list[0], marker='D', markersize=4, c='b')
        plt.plot(path_x_list[-1], path_y_list[-1], marker='D', markersize=4, c='b')

        ################## For Plotting True Map (World Coord Frame) #################
        # halfMapSize_x, halfMapSize_y = gridMapObj.worldMapSize_x/2, gridMapObj.worldMapSize_y/2
        # plt.plot([node.x for node in path], [node.y for node in path], '-ok', markersize=4)

        # # The origin of true map is at the center
        # plt.xlim([-halfMapSize_x, halfMapSize_x])
        # plt.ylim([-halfMapSize_y, halfMapSize_y])

        plt.show()


def main(gridMapFile, start, goal, epsilon, robotRadius, worldMapSize, res):
    """
    The function to run the program.

    Parameters:
        gridMapFile (str): File path of grid map file (.txt).
        start (tuple): World coordinate of start point.
        goal (tuple): World coordinate of goal point.
        epsilon (float): Step size in world-frame unit.
        robotRadius (float): Size of robot in world-frame unit.
        worldMapSize (float): Side length of world map
        res (float): Resolution of grid map.
    
    Returns:
        None
    """

    # Import a grid map
    gridMapSize = int(worldMapSize/res)
    gridMap = RRT.GridMap(gridMapFile, worldMapSize, worldMapSize, gridMapSize, res)

    print("Occupancy grid map obtained, now begin path planning...")

    # Initialize RRT
    RRTobj = RRT(start, goal, epsilon, robotRadius)
    RRTobj.obstacleList = RRTobj.getObstacleList(gridMap.gridMap)

    # Generate the path
    path = RRTobj.getPath(gridMap, epsilon, RRTobj.obstacleList, robotRadius)

    # Plot the path
    RRTobj.plotPath(gridMap, path)


if __name__ == '__main__':

    # Import grid map file
    gridMapFile = 'GridMapSample.txt'

    """
    Set required parameters (All in world-frame coordinates and units.)
    """
    # Set start and goal
    startPoint, goalPoint = (-1.6, -2.1), (1.8, 0.3)

    # Set step size
    epsilon = 0.4

    # Set robot's radius
    robotRadius = 0.1

    # Get started, have fun!
    main(gridMapFile, startPoint, goalPoint, epsilon, robotRadius, 5.0, 0.1)
    