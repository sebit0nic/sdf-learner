from SDFVisualizer import ProgressBar
import numpy as np


class SDFCurvature:
    def __init__(self, epsilon, tolerance, percentage):
        self.epsilon = epsilon
        self.tolerance = tolerance
        self.percentage = percentage

    def calculate_curvature(self, points, size):
        print('=> Computing numerical derivative and curvature of points...')
        sorted_points = []
        minima = 0
        maxima = 0
        ProgressBar.init_progress_bar()
        for z in range(size):
            for y in range(size):
                for x in range(size):
                    # Disregard points on the borders
                    if x - 1 < 0 or x + 1 >= size or y - 1 < 0 or y + 1 >= size or z - 1 < 0 or z + 1 >= size:
                        # sorted_points.append((z, y, x, 0.0))
                        # points[z][y][x].curvature = 0.0
                        continue
                    # Disregard points being nowhere near the surface
                    # if points[z][y][x].distance < -self.tolerance or points[z][y][x].distance > self.tolerance:
                    #     sorted_points.append((z, y, x, 0.0))
                    #     points[z][y][x].curvature = 0.0
                    #     continue

                    diff_sign = False
                    # Interpolate x
                    dist = points[z][y][x].distance
                    x_neg_i = dist + self.epsilon * (points[z][y][x - 1].distance - dist)
                    x_pos_i = dist + self.epsilon * (points[z][y][x + 1].distance - dist)
                    xy_pos_i = dist + self.epsilon * (points[z][y + 1][x + 1].distance - dist)
                    xy_neg_i = dist + self.epsilon * (points[z][y - 1][x - 1].distance - dist)
                    if (np.sign(dist) != np.sign(x_neg_i) or np.sign(dist) != np.sign(x_pos_i) or
                            np.sign(dist) != np.sign(xy_pos_i) or np.sign(dist) != np.sign(xy_neg_i)):
                        diff_sign = True

                    # Interpolate y
                    y_neg_i = dist + self.epsilon * (points[z][y - 1][x].distance - dist)
                    y_pos_i = dist + self.epsilon * (points[z][y + 1][x].distance - dist)
                    yz_pos_i = dist + self.epsilon * (points[z + 1][y + 1][x].distance - dist)
                    yz_neg_i = dist + self.epsilon * (points[z - 1][y - 1][x].distance - dist)
                    if (np.sign(dist) != np.sign(y_neg_i) or np.sign(dist) != np.sign(y_pos_i) or
                            np.sign(dist) != np.sign(yz_pos_i) or np.sign(dist) != np.sign(yz_neg_i)):
                        diff_sign = True

                    # Interpolate z
                    z_neg_i = dist + self.epsilon * (points[z - 1][y][x].distance - dist)
                    z_pos_i = dist + self.epsilon * (points[z + 1][y][x].distance - dist)
                    xz_pos_i = dist + self.epsilon * (points[z + 1][y][x + 1].distance - dist)
                    xz_neg_i = dist + self.epsilon * (points[z - 1][y][x - 1].distance - dist)
                    if (np.sign(dist) != np.sign(z_neg_i) or np.sign(dist) != np.sign(z_pos_i) or
                            np.sign(dist) != np.sign(xz_pos_i) or np.sign(dist) != np.sign(xz_neg_i)):
                        diff_sign = True

                    # if not diff_sign:
                    #     continue
                    # Second order derivative
                    f_dx2 = (x_pos_i - (2 * points[z][y][x].distance) + x_neg_i) / (self.epsilon ** 2)
                    f_dy2 = (y_pos_i - (2 * points[z][y][x].distance) + y_neg_i) / (self.epsilon ** 2)
                    f_dz2 = (z_pos_i - (2 * points[z][y][x].distance) + z_neg_i) / (self.epsilon ** 2)
                    f_dxy = ((xy_pos_i - x_pos_i - y_pos_i + 2 * points[z][y][x].distance - x_neg_i - y_neg_i +
                              xy_neg_i) / 2 * (self.epsilon ** 2))
                    f_dxz = ((xz_pos_i - x_pos_i - z_pos_i + 2 * points[z][y][x].distance - x_neg_i - z_neg_i +
                              xz_neg_i) / 2 * (self.epsilon ** 2))
                    f_dyz = ((yz_pos_i - y_pos_i - z_pos_i + 2 * points[z][y][x].distance - y_neg_i - z_neg_i +
                              yz_neg_i) / 2 * (self.epsilon ** 2))

                    # Curvature computation
                    curvature = (f_dx2 * (f_dy2 * f_dz2 - f_dyz * f_dxz) - f_dxy * (f_dxy * f_dz2 - f_dyz * f_dxz) +
                                 f_dxz * (f_dxy * f_dyz - f_dy2 * f_dxz))

                    if abs(curvature) < minima:
                        minima = abs(curvature)
                    if abs(curvature) > maxima:
                        maxima = abs(curvature)
                    sorted_points.append((z, y, x, curvature))
                    points[z][y][x].curvature = curvature
            ProgressBar.update_progress_bar(z / (size - 1))
        ProgressBar.end_progress_bar()
        print('   Minimum curvature found: ' + str(minima))
        print('   Maximum curvature found: ' + str(maxima))
        print('')
        return points, sorted_points

    def classify_points(self, points, sorted_points):
        print('=> Sorting curvature of points...\n')
        sorted_points.sort(key=lambda elem: abs(elem[3]), reverse=True)
        curv_list = [abs(elem[3]) for elem in sorted_points]
        s = np.array(curv_list)
        p = np.percentile(s, np.array([25, 99, 99.5]))
        print(p)
        for i in range(len(sorted_points)):
            if np.abs(sorted_points[i][3]) < p[2]:
                continue
            z = sorted_points[i][0]
            y = sorted_points[i][1]
            x = sorted_points[i][2]
            points[z][y][x].high_curvature = 1
        return points
