"""
File name: SDFUtil.py
Author: Sebastian Lackner
Version: 1.0
Description: Computation of ground truth labels out of SDF data
"""
import numpy as np

from SDFVisualizer import ProgressBar


class SDFCurvature:
    """Object used to compute points of high curvature"""

    def __init__(self, epsilon, percentage):
        self.epsilon = epsilon
        self.percentage = percentage

    def calculate_curvature(self, points, debug=True):
        """Compute gaussian curvature for each point in given array"""
        if debug:
            print('=> Computing numerical derivative and curvature of points...')
        sorted_points = []
        size = np.shape(points)[0]
        curvatures = np.zeros((size, size, size), dtype=np.float32)
        ProgressBar.init_progress_bar(debug)
        for z in range(size):
            for y in range(size):
                for x in range(size):
                    # Disregard points on the borders
                    if x - 1 < 0 or x + 1 >= size or y - 1 < 0 or y + 1 >= size or z - 1 < 0 or z + 1 >= size:
                        sorted_points.append((z, y, x, 0))
                        curvatures[z, y, x] = 0
                        continue

                    # Only points where neighbours have different signs are on the surface
                    sign = np.sign(points[z, y, x])
                    if (sign == np.sign(points[z, y, x - 1]) and sign == np.sign(points[z, y, x + 1]) and
                            sign == np.sign(points[z, y - 1, x]) and sign == np.sign(points[z, y + 1, x]) and
                            sign == np.sign(points[z - 1, y, x]) and sign == np.sign(points[z + 1, y, x]) and
                            sign == np.sign(points[z, y + 1, x + 1]) and sign == np.sign(points[z, y - 1, x - 1]) and
                            sign == np.sign(points[z + 1, y + 1, x]) and sign == np.sign(points[z - 1, y - 1, x]) and
                            sign == np.sign(points[z + 1, y, x + 1]) and sign == np.sign(points[z - 1, y, x - 1])):
                        sorted_points.append((z, y, x, 0))
                        curvatures[z, y, x] = 0
                        continue

                    # Interpolate x
                    cur_point = points[z, y, x]
                    x_neg_i = (1.0 - self.epsilon) * cur_point + self.epsilon * points[z, y, x - 1]
                    x_pos_i = (1.0 - self.epsilon) * cur_point + self.epsilon * points[z, y, x + 1]
                    x_out_1 = (1.0 - self.epsilon) * points[z, y + 1, x] + self.epsilon * points[z, y + 1, x + 1]
                    x_out_2 = (1.0 - self.epsilon) * points[z, y - 1, x] + self.epsilon * points[z, y - 1, x - 1]

                    # Interpolate y
                    y_neg_i = (1.0 - self.epsilon) * cur_point + self.epsilon * points[z, y - 1, x]
                    y_pos_i = (1.0 - self.epsilon) * cur_point + self.epsilon * points[z, y + 1, x]
                    y_out_1 = (1.0 - self.epsilon) * points[z + 1, y, x] + self.epsilon * points[z + 1, y + 1, x]
                    y_out_2 = (1.0 - self.epsilon) * points[z - 1, y, x] + self.epsilon * points[z - 1, y - 1, x]

                    # Interpolate z
                    z_neg_i = (1.0 - self.epsilon) * cur_point + self.epsilon * points[z - 1, y, x]
                    z_pos_i = (1.0 - self.epsilon) * cur_point + self.epsilon * points[z + 1, y, x]
                    z_out_1 = (1.0 - self.epsilon) * points[z, y, x + 1] + self.epsilon * points[z + 1, y, x + 1]
                    z_out_2 = (1.0 - self.epsilon) * points[z, y, x - 1] + self.epsilon * points[z - 1, y, x - 1]

                    # Bilinear interpolate for mixed coordinates
                    xy_pos_i = (1.0 - self.epsilon) * x_pos_i + self.epsilon * x_out_1
                    xy_neg_i = (1.0 - self.epsilon) * x_neg_i + self.epsilon * x_out_2
                    yz_pos_i = (1.0 - self.epsilon) * y_pos_i + self.epsilon * y_out_1
                    yz_neg_i = (1.0 - self.epsilon) * y_neg_i + self.epsilon * y_out_2
                    xz_pos_i = (1.0 - self.epsilon) * z_pos_i + self.epsilon * z_out_1
                    xz_neg_i = (1.0 - self.epsilon) * z_neg_i + self.epsilon * z_out_2

                    # Second order derivatives
                    f_dx = (x_pos_i - x_neg_i) / (2.0 * self.epsilon)
                    f_dy = (y_pos_i - y_neg_i) / (2.0 * self.epsilon)
                    f_dz = (z_pos_i - z_neg_i) / (2.0 * self.epsilon)
                    f_dx2 = (x_pos_i - (2.0 * cur_point) + x_neg_i) / (self.epsilon ** 2)
                    f_dy2 = (y_pos_i - (2.0 * cur_point) + y_neg_i) / (self.epsilon ** 2)
                    f_dz2 = (z_pos_i - (2.0 * cur_point) + z_neg_i) / (self.epsilon ** 2)
                    f_dxy = ((xy_pos_i - x_pos_i - y_pos_i + 2.0 * cur_point - x_neg_i - y_neg_i +
                              xy_neg_i) / 2.0 * (self.epsilon ** 2))
                    f_dxz = ((xz_pos_i - x_pos_i - z_pos_i + 2.0 * cur_point - x_neg_i - z_neg_i +
                              xz_neg_i) / 2.0 * (self.epsilon ** 2))
                    f_dyz = ((yz_pos_i - y_pos_i - z_pos_i + 2.0 * cur_point - y_neg_i - z_neg_i +
                              yz_neg_i) / 2.0 * (self.epsilon ** 2))

                    ext_hessian = np.array([[f_dx2, f_dxy, f_dxz, f_dx], [f_dxy, f_dy2, f_dyz, f_dy],
                                            [f_dxz, f_dyz, f_dz2, f_dz], [f_dx, f_dy, f_dz, 0.0]])

                    # Curvature computation
                    curvature = - np.linalg.det(ext_hessian) / ((f_dx ** 2 + f_dy ** 2 + f_dz ** 2) ** 2)

                    sorted_points.append((z, y, x, curvature))
                    curvatures[z, y, x] = curvature
            ProgressBar.update_progress_bar(debug, z / (size - 1))
        ProgressBar.end_progress_bar(debug)
        return curvatures, sorted_points

    def classify_points(self, points, sorted_points, debug=True):
        """Assign class to each point based on sorted curvature, meaning to find points of high curvature"""
        if debug:
            print('=> Sorting curvature of points...')
        sorted_points.sort(key=lambda elem: abs(elem[3]), reverse=True)
        size = np.shape(points)[0]
        points_of_interest = np.zeros((size, size, size), dtype=np.int32)

        # Assign positive class 1 to points of high curvature up to certain percentage
        for i in range(0, int(len(sorted_points) * (self.percentage / 100.0))):
            z = sorted_points[i][0]
            y = sorted_points[i][1]
            x = sorted_points[i][2]
            points_of_interest[z, y, x] = 1
        return points_of_interest
