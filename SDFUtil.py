from SDFVisualizer import ProgressBar
import numpy as np


class SDFCurvature:
    def __init__(self, epsilon, tolerance, percentage):
        self.epsilon = epsilon
        self.tolerance = tolerance
        self.percentage = percentage

    def calculate_curvature(self, points, debug=True):
        if debug:
            print('=> Computing numerical derivative and curvature of points...')
        sorted_points = []
        minima = 0
        maxima = 0
        size = np.shape(points)[0]
        curvatures = np.zeros((size, size, size), dtype=np.float32)
        ProgressBar.init_progress_bar(debug)
        for z in range(size):
            for y in range(size):
                for x in range(size):
                    # Disregard points on the borders
                    if x - 1 < 0 or x + 1 >= size or y - 1 < 0 or y + 1 >= size or z - 1 < 0 or z + 1 >= size:
                        continue

                    # Interpolate x
                    dist = points[z, y, x]
                    x_neg_i = dist + self.epsilon * (points[z, y, x - 1] - dist)
                    x_pos_i = dist + self.epsilon * (points[z, y, x + 1] - dist)
                    xy_pos_i = dist + self.epsilon * (points[z, y + 1, x + 1] - dist)
                    xy_neg_i = dist + self.epsilon * (points[z, y - 1, x - 1] - dist)

                    # Interpolate y
                    y_neg_i = dist + self.epsilon * (points[z, y - 1, x] - dist)
                    y_pos_i = dist + self.epsilon * (points[z, y + 1, x] - dist)
                    yz_pos_i = dist + self.epsilon * (points[z + 1, y + 1, x] - dist)
                    yz_neg_i = dist + self.epsilon * (points[z - 1, y - 1, x] - dist)

                    # Interpolate z
                    z_neg_i = dist + self.epsilon * (points[z - 1, y, x] - dist)
                    z_pos_i = dist + self.epsilon * (points[z + 1, y, x] - dist)
                    xz_pos_i = dist + self.epsilon * (points[z + 1, y, x + 1] - dist)
                    xz_neg_i = dist + self.epsilon * (points[z - 1, y, x - 1] - dist)

                    # Second order derivative
                    f_dx2 = (x_pos_i - (2 * points[z, y, x]) + x_neg_i) / (self.epsilon ** 2)
                    f_dy2 = (y_pos_i - (2 * points[z, y, x]) + y_neg_i) / (self.epsilon ** 2)
                    f_dz2 = (z_pos_i - (2 * points[z, y, x]) + z_neg_i) / (self.epsilon ** 2)
                    f_dxy = ((xy_pos_i - x_pos_i - y_pos_i + 2 * points[z, y, x] - x_neg_i - y_neg_i +
                              xy_neg_i) / 2 * (self.epsilon ** 2))
                    f_dxz = ((xz_pos_i - x_pos_i - z_pos_i + 2 * points[z, y, x] - x_neg_i - z_neg_i +
                              xz_neg_i) / 2 * (self.epsilon ** 2))
                    f_dyz = ((yz_pos_i - y_pos_i - z_pos_i + 2 * points[z, y, x] - y_neg_i - z_neg_i +
                              yz_neg_i) / 2 * (self.epsilon ** 2))

                    # Curvature computation
                    curvature = (f_dx2 * (f_dy2 * f_dz2 - f_dyz * f_dxz) - f_dxy * (f_dxy * f_dz2 - f_dyz * f_dxz) +
                                 f_dxz * (f_dxy * f_dyz - f_dy2 * f_dxz))

                    if abs(curvature) < minima:
                        minima = abs(curvature)
                    if abs(curvature) > maxima:
                        maxima = abs(curvature)
                    sorted_points.append((z, y, x, curvature))
                    curvatures[z, y, x] = curvature
            ProgressBar.update_progress_bar(debug, z / (size - 1))
        ProgressBar.end_progress_bar(debug)
        if debug:
            print('   Minimum curvature found: ' + str(minima))
            print('   Maximum curvature found: ' + str(maxima))
            print('')
        return curvatures, sorted_points

    def classify_points(self, points, sorted_points, debug=True):
        if debug:
            print('=> Sorting curvature of points...')
        sorted_points.sort(key=lambda elem: abs(elem[3]), reverse=True)
        curv_list = [abs(elem[3]) for elem in sorted_points]
        size = np.shape(points)[0]
        points_of_interest = np.zeros((size, size, size), dtype=np.int32)
        s = np.array(curv_list)
        p = np.percentile(s, np.array([25, 98.5, 99]))
        if debug:
            print(f'   Percentiles: {p}')
            print('')
        for i in range(len(sorted_points)):
            if p[1] < np.abs(sorted_points[i][3]) < p[2]:
                z = sorted_points[i][0]
                y = sorted_points[i][1]
                x = sorted_points[i][2]
                points_of_interest[z, y, x] = 1
        return points_of_interest
