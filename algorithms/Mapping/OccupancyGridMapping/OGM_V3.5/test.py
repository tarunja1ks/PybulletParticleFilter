from Path_Sim import Simulation
import matplotlib.pyplot as plt
import numpy as np
from math import *
from decimal import *
from IPython.display import clear_output
from time import time
from util import World2Grid, Grid2World

class OGM():
    def __init__(self, res, log_prior):
        self.sim = Simulation()
        self.res = res
        self.log_prior = log_prior
        self.trueMapSize = self.sim.sim.get_env_info()["map_size"]
        self.gridMapSize = int(self.trueMapSize/self.res)
        self.log_t = np.zeros((self.gridMapSize, self.gridMapSize))
        # Initialize the probabilistic grid map
        self.probGridMap = np.ones((self.log_t.shape[0], self.log_t.shape[1])) * 0.5


    def mapping(self, dataset, xt_start_grid) -> np.ndarray:
        self.probGridMap = np.ones((self.log_t.shape[0], self.log_t.shape[1])) * 0.5

        # Extract data at time t
        pose, hitPoints = dataset[0], dataset[1]

        # Update log odds
        # self.update_log_odds(hitPoints, pose)

        rayConeGrids_World, Zt, measPhi, endPoints = self.perceptedCells(pose, hitPoints)

        # # Update the log odds for all cells with the perceptual field of lidar
        for grid in rayConeGrids_World:
            grid_coord = World2Grid(grid, self.gridMapSize, self.res)
            self.probGridMap[grid_coord[1]][grid_coord[0]] = 1

        for point in endPoints:
            grid = World2Grid(point, self.gridMapSize, self.res)
            self.probGridMap[grid[1]][grid[0]] = 0
        
        # start = World2Grid(pose, self.gridMapSize, self.res)
        # self.probGridMap[start[0]][start[1]] = 0

        self.probGridMap[xt_start_grid[1]][xt_start_grid[0]] = 0
        
        # Update the probabilistic grid map based on the latest log odds
        # self.updateProbGridMap()


    def bresenham(self, x_start, y_start, x_end, y_end):
        # Normalize the grid side length to 1
        scale = Decimal(str(1.0)) / Decimal(str(self.res))
        x_start, x_end = Decimal(str(x_start))*scale, Decimal(str(x_end))*scale
        y_start, y_end = Decimal(str(y_start))*scale, Decimal(str(y_end))*scale
        
        # Check if slope > 1
        dy0 = y_end - y_start
        dx0 = x_end - x_start
        steep = abs(dy0) > abs(dx0)
        if steep:
            x_start, y_start = y_start, x_start
            x_end, y_end = y_end, x_end

        # Change direction if x_start > x_end
        if x_start > x_end:
            x_start, x_end = x_end, x_start
            y_start, y_end = y_end, y_start

        # Determine the moving direction for y
        if y_start <= y_end:
            y_step = Decimal(str(1))
        else:
            y_step = Decimal(str(-1))
        
        dx = x_end - x_start
        dy = abs(y_end - y_start)
        error = Decimal(str(0.0))
        d_error = dy / dx
        step = Decimal(str(1.0))
        yk = y_start

        perceptedGrids = []

        # Iterate over the grids from the adjusted x_start to x_end
        for xk in np.arange(x_start, x_end+step, step):
            if steep:
                new_x = yk
                new_y = xk
            else:
                new_x = xk
                new_y = yk
            
            # Scale back to the original resolution and append to the list
            perceptedGrids.append((float(new_x/scale), float(new_y/scale)))
            # perceptedGrids.append((float(new_x), float(new_y)))

            error += d_error
            if error >= 0.5:
                yk += y_step
                error -= step

        return np.array(perceptedGrids)


    def perceptedCells(self, xt, hitPoints):
        rayConeGrids_World = np.array([(0, 0)])
        Zt = np.zeros(hitPoints.shape[0])
        measPhi = np.zeros(hitPoints.shape[0])

        endPoints = []

        # Iterate thru each ray to collect data
        for i in range(hitPoints.shape[0]):
            point = hitPoints[i]

            # When there's no hit for a ray, determine its end point
            if np.isnan(point[0]):
                # Relative angle between robot and the ray (Range: RAY_START_ANG - RAY_END_ANG)
                theta_body = Decimal(str(self.sim.rayStartAng)) \
                    + (Decimal(str(self.sim.beta)) * Decimal(str(i)))
                # Convert to range: +- (RAY_END_ANG - RAY_START_ANG)/2 [positive: ccw]
                ray_robot_ang = Decimal(str(pi/2)) - theta_body

                x0 = Decimal(str(self.sim.Z_max)) \
                    * Decimal(str(cos(ray_robot_ang + Decimal(str(xt[2]))))) + Decimal(str(xt[0]))
                y0 = Decimal(str(self.sim.Z_max)) \
                    * Decimal(str(sin(ray_robot_ang + Decimal(str(xt[2]))))) + Decimal(str(xt[1]))
                point = (float(x0), float(y0))

            # Discard the invalid point
            if (abs(point[0]) >= self.trueMapSize/2) or (abs(point[1]) >= self.trueMapSize/2):
                continue

            Zt[i] = sqrt((point[0]-xt[0])**2 + (point[1]-xt[1])**2)
            measPhi[i] = atan2(point[1]-xt[1], point[0]-xt[0]) - xt[2]
            
            # Determine the starting and end grid centers in world coordinates
            # xt_grid = World2Grid(xt, self.gridMapSize, self.res)
            # xtGridWorld = Grid2World(xt_grid, self.gridMapSize, self.res)
            # end_grid = World2Grid(point, self.gridMapSize, self.res)
            # endGridWorld = Grid2World(end_grid, self.gridMapSize, self.res)

            # # Determine the grids passed by the ray
            # ray_grids = self.bresenham(xtGridWorld[0],
            #                            xtGridWorld[1],
            #                            endGridWorld[0],
            #                            endGridWorld[1])
            ray_grids = self.bresenham(xt[0],
                                       xt[1],
                                       point[0],
                                       point[1])
            endPoints.append(point)
            
            # for grid in ray_grids:
            #     if (abs(grid[0]) > self.trueMapSize) or (abs(grid[1]) > self.trueMapSize):
            #         print("invalid ray_grids:", ray_grids, "xt:", xt, "end point:", point)
            #         continue
            
            rayConeGrids_World = np.concatenate((rayConeGrids_World, ray_grids),
                                                 axis=0)
        
        rayConeGrids_World = np.unique(rayConeGrids_World[1:], axis=0)

        return rayConeGrids_World, Zt, measPhi, np.array(endPoints)


    def inv_sensor_model(self, xt, grid_mi, Zt, Z_max, measPhi):
        # Set log-odds values
        l_occ = log(0.55/0.45)
        l_free = log(0.45/0.55)

        # Distance between robot and grid center
        r = sqrt((grid_mi[0]-xt[0])**2 + (grid_mi[1]-xt[1])**2)
        # Relative angle between robot and grid center
        phi = atan2(grid_mi[1]-xt[1], grid_mi[0]-xt[0]) - xt[2]
        # Index of the ray that corresponds to this measurement
        k = np.argmin(abs(np.subtract(phi, measPhi)))

        # Determine the update of log odd score for this grid
        if ((r > np.minimum(Z_max, Zt[k]+self.res/2.0)) or 
            (np.abs(phi-measPhi[k]) > self.sim.beta/2.0)):
            
            return self.log_prior

        elif ((Zt[k] < Z_max) and (np.abs(r-Zt[k]) < self.res/2.0*sqrt(2))):
            return l_occ

        elif (r < Zt[k]):
            return l_free
        

    def update_log_odds(self, hitPoints, xt):
        rayConeGrids_World, Zt, measPhi, endPoints = self.perceptedCells(xt, hitPoints)

        # Update the log odds for all cells with the perceptual field of lidar
        for grid in rayConeGrids_World:
            grid_coord = World2Grid(grid, self.gridMapSize, self.res)
            self.probGridMap[grid_coord[0]][grid_coord[1]] = 1

        for point in endPoints:
            grid = World2Grid(point, self.gridMapSize, self.res)
            self.probGridMap[grid[0]][grid[1]] = 0
        start = World2Grid(xt, self.gridMapSize, self.res)
        self.probGridMap[start[0]][start[1]] = 0

            # try:
            #     self.log_t[grid_coord[0]][grid_coord[1]] += \
            #         self.inv_sensor_model(xt, grid, Zt, self.sim.Z_max, measPhi) \
            #         - self.log_prior
            # except:
            #     pass

    def updateProbGridMap(self):
        # Convert log odds to probabilities and set the occupancy status of grids
        for i in range(self.log_t.shape[0]):
            for j in range(self.log_t.shape[1]):

                P_mi = 1 - 1/(1+np.exp(self.log_t[i][j]))

                # When the grid is likely to be occupied
                if (P_mi > 0.5):
                    self.probGridMap[i][j] = 0 # set to zero for plotting in black

                # When the grid's status is likely undetermined
                elif (P_mi == 0.5):
                    self.probGridMap[i][j] = 0.5 # set to 0.5 for plotting in grey

                # When the grid is likely to be free
                else:
                    self.probGridMap[i][j] = 1 # set to one for plotting in white


def main(resolution=0.1, log_prior=0.0):
    t0 = time()
    
    ogm = OGM(resolution, log_prior)
    fig = plt.figure()

    while True:
        # Get latest data
        trueMap, dataset, status = ogm.sim.collectData()
        carPos_grid = World2Grid(dataset[0], ogm.gridMapSize, ogm.res)
        pos_true = Grid2World(carPos_grid, ogm.gridMapSize, ogm.res)
        ogm.mapping(dataset, carPos_grid)

        if status == -1:
            break
        
        # Display real-time true map and occupancy grid map
        if trueMap is not None:
            plt.clf()
            clear_output(wait=True)

            fig.add_subplot(1, 2, 1)
            plt.imshow(trueMap)
            # plt.scatter(dataset[0][0], dataset[0][1], s=40, marker='s', c='y')
            # plt.scatter(pos_true[0], pos_true[1], s=40, marker='o', c='g')
            # plt.xlim([-2.5, 2.5])
            # plt.ylim([-2.5, 2.5])
            # plt.grid()
            plt.title('Physical World')

            fig.add_subplot(1, 2, 2)
            plt.imshow(ogm.probGridMap, cmap='gray', vmin = 0, vmax = 1, origin='lower')
            # plt.scatter(carPos_grid[0], carPos_grid[1], s=80, marker='o', c='b', edgecolors='r')
            plt.title('Real-time Grid Map')
            
            plt.show(block=False)
            plt.pause(0.001)


if __name__ == '__main__':
    main()

    # fig = plt.figure()
    # res = 0.1
    # grid_map = np.ones((50,50))

    # for i in range(grid_map.shape[0]):
    #     for j in range(grid_map.shape[1]):
    #         grid_map[i][j] = 0
    #         world = Grid2World((j,i))
    #         grid = World2Grid(world, 50)

    #         plt.clf()
    #         fig.add_subplot(1, 2, 1)
    #         plt.scatter(world[0], world[1], s=40, marker='s', c='g')
    #         plt.title('Physical World')
    #         plt.xlim([-2.5, 2.5])
    #         plt.ylim([-2.5, 2.5])
    #         plt.grid()

    #         fig.add_subplot(1, 2, 2)
    #         plt.imshow(grid_map, cmap='gray', vmin = 0, vmax = 1, origin='lower')
    #         plt.scatter(grid[0], grid[1], s=20, marker='o', c='g')
    #         plt.title('Grid Map')

    #         plt.show(block=False)
    #         plt.pause(0.001)

            # grid_map[i][j] = 1
