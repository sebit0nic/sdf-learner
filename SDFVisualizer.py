"""
File name: SDFVisualizer.py
Author: Sebastian Lackner
Version: 1.0
Description: Visualization of input data and progress
"""

import numpy as np
import pyvista


class ProgressBar:
    """Object used to compute the progress of some task"""

    @staticmethod
    def init_progress_bar(debug):
        """Initialize progress bar at the start of task"""
        if not debug:
            return
        print('   Progress: ' + 100 * '.', end='')

    @staticmethod
    def update_progress_bar(debug, progress):
        """Update progress to visualize task status"""
        if not debug:
            return
        print('\r   Progress: ' + (int(progress * 100) * '#') + (100 - int(progress * 100)) * '.', end='', flush=True)

    @staticmethod
    def end_progress_bar(debug):
        """Show final progress bar at the end of task"""
        if not debug:
            return
        print('\r   Progress: ' + 100 * '#', end='', flush=True)
        print('')


class SDFVisualizer:
    """Object used to visualize all sorts of data"""

    def __init__(self, point_size):
        self.point_size = point_size

    def plot_points(self, points_of_interest, i_obj_file):
        """Plot point cloud out of SDF, prediction, or ground truth data"""
        arr_curv = []
        size_p = np.shape(points_of_interest)[0]
        print('=> Visualizing points using pyvista...')
        print('   Progress: ' + 100 * '.', end='')
        for z in range(size_p):
            for y in range(size_p):
                for x in range(size_p):
                    # Add point with high curvature to blue points being plotted
                    if points_of_interest[z, y, x]:
                        arr_curv.append((float(z + 0.5), float(y + 0.5), float(x + 0.5)))
            print('\r   Progress: ' + (int((z / (size_p - 1)) * 100) * '#') + (100 - int((z / (size_p - 1)) * 100)) *
                  '.', end='', flush=True)
        print('')

        # Plot mesh as reference (if it exists) + points of high curvature
        plotter = pyvista.Plotter()
        try:
            plotter.import_obj(i_obj_file)
        except FileNotFoundError:
            print('Obj not found for sample !')
        if len(arr_curv) != 0:
            pc_curv = np.array(arr_curv)
            plotter.add_mesh(pc_curv, color='b', point_size=self.point_size, render_points_as_spheres=True, opacity=1)
        plotter.show_axes()
        plotter.show()
