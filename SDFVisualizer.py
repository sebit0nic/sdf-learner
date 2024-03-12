import numpy as np
import pyvista


class ProgressBar:
    @staticmethod
    def init_progress_bar():
        print('   Progress: ' + 100 * '.', end='')

    @staticmethod
    def update_progress_bar(progress):
        print('\r   Progress: ' + (int(progress * 100) * '#') + (100 - int(progress * 100)) * '.', end='', flush=True)

    @staticmethod
    def end_progress_bar():
        print('\r   Progress: ' + 100 * '#', end='', flush=True)
        print('')


class SDFVisualizer:
    def __init__(self, point_size):
        self.point_size = point_size

    def plot_points(self, points, size):
        arr_in = []
        arr_curv_pos = []
        arr_curv_neg = []
        print('=> Visualizing points using pyvista...')
        print('   Progress: ' + 100 * '.', end='')
        for z in range(size):
            for y in range(size):
                for x in range(size):
                    if points[z][y][x].high_curvature:
                        if points[z][y][x].curvature > 0:
                            arr_curv_pos.append((float(x), float(y), float(z)))
                        else:
                            arr_curv_neg.append((float(x), float(y), float(z)))
                    elif points[z][y][x].distance <= 0:
                        arr_in.append((float(x), float(y), float(z)))
            print('\r   Progress: ' + (int((z / (size - 1)) * 100) * '#') + (100 - int((z / (size - 1)) * 100)) * '.',
                  end='', flush=True)
        print('')
        plotter = pyvista.Plotter()
        plotter.add_mesh(pyvista.Box(bounds=(0.0, size, 0.0, size, 0.0, size)), color='red', opacity=0.01)
        if len(arr_in) != 0:
            pc_in = np.array(arr_in)
            plotter.add_mesh(pc_in, color='g', point_size=self.point_size, render_points_as_spheres=True, opacity=1)
        if len(arr_curv_pos) != 0:
            pc_curv = np.array(arr_curv_pos)
            plotter.add_mesh(pc_curv, color='y', point_size=self.point_size, render_points_as_spheres=True, opacity=1)
        if len(arr_curv_neg) != 0:
            pc_curv = np.array(arr_curv_neg)
            plotter.add_mesh(pc_curv, color='b', point_size=self.point_size, render_points_as_spheres=True, opacity=1)
        plotter.show_axes()
        plotter.show_grid()
        plotter.show()
