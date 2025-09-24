# ------------------------------------------------------------------
# Mapping and Path Planning Integration
# | Status: COMPLETED (Version: 1)
# | Contributors: Jeffrey
#
# Function:
# Do all by one click: run the simulation, do occupancy grid mapping, and
# obtain a path from start to goal via RRT.
# ------------------------------------------------------------------

import OccupancyGridMapping.Occupancy_Grid_Mapping as OGM
import RRT.RRT

def main(startPoint, goalPoint):
    """
    The function to run the simulation, do occupancy grid mapping, and
        obtain a path from start to goal via RRT.

    Parameters:
        startPoint (tuple): Start point in world coordinate.
        goalPoint (tuple): Goal in world coordinate.

    Returns:
        None
    """

    # Set parameters for OGM
    resolution, log_prior, save_grid_map = 0.1, 0.0, True
    filename = "probGridMap.txt"

    # Mapping
    OGM.main(resolution, log_prior, save_grid_map)

    # Set parameters for RRT
    epsilon, robotRadius, worldMapSize = 0.5, 0.1, 10

    # Generate path
    RRT.main(filename, startPoint, goalPoint, epsilon, robotRadius, worldMapSize, resolution)

if __name__ == '__main__':
    start, goal = (0.0, 0.0), (4.5, 4.5)
    main(start, goal)
    