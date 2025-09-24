#!/usr/bin/env python
"""
Extract Occupancy Grid Map as .npz

---

UCSD ERL Y. Yi, v1.0

---
"""
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


class GridMapConversion:
    """
    This class contains two utility functions doing coordinate conversion
    """
    def __init__(self, grid_origin=np.array([0, 0]), grid_res=0.1):
        self._grid_origin = grid_origin
        self._grid_res = grid_res

    def meter2cell(self, loc, debug=False):
        """
        Convert environment from meter to cell
        Input:
            loc:     nd vector containing loc coordinates in world frame (in meters)
            grid_min: nd vector containing the lower left corner of the grid (in meters)\n"
            grid_res: nd vector containing the grid resolution (in meters)\n"
        Output:
            loc_cell: nd vector containing index of cell grid index goes
            from lower left to upper right
        """
        grid_min = self._grid_origin
        grid_res = self._grid_res
        loc = np.array(loc)
        try:
            diff = (loc - grid_min) / grid_res
            loc_cell = diff.astype(int)
        except FloatingPointError:
            print('grid_map_dist_err GRID RES:', grid_res)
            print('grid_map_dist_err GRID MIN:', grid_min)
            print('grid_map_dist_err LOC:', loc)

        if debug:
            print("loc at [%.2f %.2f], loc cell at [%d, %d]" % (loc[0], loc[1], loc_cell[0], loc_cell[1]))
        return loc_cell

    def cell2meter(self, loc_cell, debug=False):
        """
        Input:
            cell_loc: nd vector containing the cell index (from lower left to upper right)
            mesh_origin: nd vector containing the cell origin (in meters)"
            grid_res: nd vector containing the grid resolution (in meters)\n"
        Output:
            loc: nd vector containing loc coordinates in world frame (in meters)
            from lower left to upper right
        """
        grid_res = self._grid_res
        loc = np.array(loc_cell) * grid_res + self._grid_origin

        if debug:
            print("loc cell at [%d,  %d], loc at [%.2f, %.2f]" % (loc_cell[0], loc_cell[1], loc[0], loc[1]))
        return loc


def grid_map_plotter(ax, pickle_dir):
    """
    Load Jackal_Race Grid Map for Pickle File and plot it as scatter
    """
    res_dict = pkl.load(open(pickle_dir, "rb"), encoding='latin1')

    # map decoding
    map_raw_data = np.array(res_dict['map_data'], dtype=np.int8)
    map_res = res_dict["map_resolution"]
    map_height = res_dict["map_height"]
    map_width = res_dict["map_width"]

    # Convert 2d cell location information into meters
    grid_x_axis = np.linspace(0, map_res * map_width, map_width) - (map_width - 1) / 2 * map_res
    grid_y_axis = np.linspace(0, map_res * map_height, map_height) - (map_height - 1) / 2 * map_res

    grid_map_conversion = GridMapConversion(grid_origin=np.array([grid_x_axis[0],
                                                                  grid_y_axis[0]]),
                                            grid_res=map_res)

    # Convert raw map data into matrix
    map_raw_mat = map_raw_data.reshape(map_width, map_height).T

    # only legend once
    first_point = True

    # Validate the matrix conversion is correct
    for i in range(map_width):
        for j in range(map_height):
            if map_raw_mat[i, j] >= 80:
                if first_point:
                    ax.scatter(grid_x_axis[i], grid_y_axis[j], c='grey', marker='s', s=20, label='Obstacles')
                    first_point = False
                else:
                    ax.scatter(grid_x_axis[i], grid_y_axis[j], c='grey', marker='s', s=20)

    # # Validate the coordinate conversion meter to cell is correct
    # for x in grid_x_axis[0:-2]:
    #     for y in grid_y_axis[0:-2]:
    #         loc = grid_map_conversion.meter2cell(np.array([x, y]))
    #
    #         if map_raw_mat[loc[0], loc[1]] >= 80:
    #             if first_point:
    #                 # again, only legend once
    #                 ax.scatter(x, y, c='grey', marker='s', s=20, label='Obstacles')
    #                 first_point = False
    #             else:
    #                 ax.scatter(x, y, c='grey', marker='s', s=20)

    # # Validate the coordinate conversion cell to meter is correct
    # for i in range(map_width):
    #     for j in range(map_height):
    #         xy_coordinate = grid_map_conversion.cell2meter(np.array([i, j]))
    #         x, y = xy_coordinate[0], xy_coordinate[1]
    #
    #         if map_raw_mat[i, j] >= 80:
    #             if first_point:
    #                 # again, only legend once
    #                 ax.scatter(x, y, c='grey', marker='s', s=20, label='Obstacles')
    #                 first_point = False
    #             else:
    #                 ax.scatter(x, y, c='grey', marker='s', s=20)


if __name__ == "__main__":

    pkl_name = "jackal_race_ogm.pkl"
    pkl_dir = os.path.join(os.getcwd(), pkl_name)

    # Create fig
    fig_1 = plt.figure(figsize=(8, 6))
    fig_1_ax1 = fig_1.add_subplot(111)
    grid_map_plotter(fig_1_ax1, pkl_dir)
    plt.show(block=True)
