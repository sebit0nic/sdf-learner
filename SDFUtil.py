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

        # curv_list = [abs(elem[3]) for elem in sorted_points]
        # s = np.array(curv_list)
        # p = np.percentile(s, np.array([90, 100]))

        # Assign positive class 1 to points of high curvature up to certain percentage
        # for i in range(len(sorted_points)):
        #     if p[0] <= np.abs(sorted_points[i][3]) <= p[1]:
        for i in range(0, int(len(sorted_points) * (self.percentage / 100.0))):
            z = sorted_points[i][0]
            y = sorted_points[i][1]
            x = sorted_points[i][2]
            points_of_interest[z, y, x] = 1
        return points_of_interest
