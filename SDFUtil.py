"""
File name: SDFUtil.py
Author: Sebastian Lackner
Version: 1.0
Description: Computation of ground truth labels out of SDF data
"""
import numpy as np
import math

from SDFVisualizer import ProgressBar


class SDFCurvature:
    """Object used to compute points of high curvature"""

    def __init__(self, epsilon, percentage, size):
        self.epsilon = epsilon
        self.percentage = percentage
        self.size = size

    def calculate_curvature(self, sdf, debug=True):
        """Compute gaussian curvature for each point in given array"""
        if debug:
            print('=> Computing numerical derivative and curvature of points...')
        sorted_points = []
        curvatures = np.zeros((self.size, self.size, self.size), dtype=np.float32)
        neg_curvatures = []
        pos_curvatures = []
        ProgressBar.init_progress_bar(debug)
        for z in range(self.size):
            for y in range(self.size):
                for x in range(self.size):
                    # Disregard points that are not on the surface
                    if not sdf.on_surface(np.array((z, y, x)))[0]:
                        continue

                    # Compute the normal and hessian at the current point
                    normal = sdf.surface_normal(np.array((z, y, x)))
                    hessian = sdf.curvature(np.array((z, y, x)))

                    # Compute the shape operator, also called the weingarten map
                    weingarten_map = -hessian * normal
                    res = np.linalg.det(weingarten_map)

                    sorted_points.append((z, y, x, abs(res)))
                    curvatures[z, y, x] = res
                    if res >= 0:
                        pos_curvatures.append(res)
                    else:
                        neg_curvatures.append(res)
            ProgressBar.update_progress_bar(debug, z / (self.size - 1))
        ProgressBar.end_progress_bar(debug)
        print(f'   Min pos: {min(pos_curvatures)}, max pos: {max(pos_curvatures)}')
        print(f'   Min neg: {min(neg_curvatures)}, max neg: {max(neg_curvatures)}')
        return curvatures, sorted_points

    def classify_points(self, sorted_points, debug=True):
        """Assign class to each point based on sorted curvature, meaning to find points of high curvature"""
        if debug:
            print('=> Sorting curvature of points...')
        sorted_points.sort(key=lambda elem: elem[3], reverse=True)
        points_of_interest = np.zeros((self.size, self.size, self.size), dtype=np.int32)

        for i in range(0, int(len(sorted_points) * (self.percentage / 100.0))):
            z = sorted_points[i][0]
            y = sorted_points[i][1]
            x = sorted_points[i][2]
            points_of_interest[z, y, x] = 1
        return points_of_interest

    def classify_points_from_mesh(self, curvature_data, distances, debug=True):
        """Assign class to each point based on vertices with high curvature in the neighbourhood"""
        if debug:
            print('=> Computing points of high curvature from mesh...')

        # Sort curvatures of vertices based on absolute value in descending order to later iterate over them
        sorted_curvature = curvature_data[curvature_data[:, 3].argsort()[::-1]]

        # Classify a certain percentage of vertices of the mesh as high curvature points in the SDF
        num_curvatures = round(np.shape(sorted_curvature)[0] * (self.percentage / 100))
        points_of_interest = np.zeros((self.size, self.size, self.size), dtype=np.int32)
        curvatures = np.zeros((self.size, self.size, self.size))
        for i in range(num_curvatures):
            v = sorted_curvature[i]

            # Get coordinates of neighbouring points in SDF based on mesh vertex coordinate
            f0 = math.floor(v[0])
            c0 = math.ceil(v[0])
            f1 = math.floor(v[1])
            c1 = math.ceil(v[1])
            f2 = math.floor(v[2])
            c2 = math.ceil(v[2])
            cur_dist = math.inf
            cur_z = 0
            cur_y = 0
            cur_x = 0
            loc_array = [[f0, f1, f2], [f0, f1, c2], [f0, c1, f2], [f0, c1, c2], [c0, f1, f2], [c0, f1, c2],
                         [c0, c1, f2], [c0, c1, c2]]
            dist_array = [distances[f0, f1, f2], distances[f0, f1, c2], distances[f0, c1, f2], distances[f0, c1, c2],
                          distances[c0, f1, f2], distances[c0, f1, c2], distances[c0, c1, f2], distances[c0, c1, c2]]

            # Find neighbouring SDF point with the smallest distance to surface to place point as close as possible
            for k in range(len(loc_array)):
                if abs(dist_array[k]) < cur_dist:
                    cur_dist = abs(dist_array[k])
                    cur_z = loc_array[k][0]
                    cur_y = loc_array[k][1]
                    cur_x = loc_array[k][2]

            # Classify corresponding point in SDF as high curvature point
            points_of_interest[cur_z, cur_y, cur_x] = 1
            curvatures[cur_z, cur_y, cur_x] = sorted_curvature[i][4]
        return points_of_interest, curvatures
