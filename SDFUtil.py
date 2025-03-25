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

    def __init__(self, epsilon, threshold, percentage):
        self.epsilon = epsilon
        self.threshold = threshold
        self.percentage = percentage

    def hessian(self, x):
        """
        Calculate the hessian matrix with finite differences
        Parameters:
           - x : ndarray
        Returns:
           an array of shape (x.dim, x.ndim) + x.shape
           where the array[i, j, ...] corresponds to the second derivative x_ij
        """
        # Reference: https://stackoverflow.com/a/31207520
        x_grad = np.gradient(x, self.epsilon)
        hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
        for k, grad_k in enumerate(x_grad):
            # iterate over dimensions, apply gradient again to every component of the first derivative
            tmp_grad = np.gradient(grad_k, self.epsilon)
            for l, grad_kl in enumerate(tmp_grad):
                hessian[k, l, :, :] = grad_kl
        return hessian

    def calculate_curvature(self, points, debug=True):
        """Compute gaussian curvature for each point in given array"""
        if debug:
            print('=> Computing numerical derivative and curvature of points...')
        sorted_points = []
        size = np.shape(points)[0]
        curvatures = np.zeros((size, size, size), dtype=np.float32)
        gradients = np.gradient(points)
        hessians = self.hessian(points)
        ProgressBar.init_progress_bar(debug)
        for z in range(size):
            for y in range(size):
                for x in range(size):
                    # Disregard points that are not near the surface
                    if points[z, y, x] < -self.threshold or points[z, y, x] > self.threshold:
                        sorted_points.append((z, y, x, 0))
                        curvatures[z, y, x] = 0
                        continue

                    # Compute expanded hessian and norm of gradient
                    gradient = np.array([gradients[0][z, y, x], gradients[1][z, y, x], gradients[2][z, y, x]])
                    hessian = hessians[:, :, z, y, x]
                    ext_hessian = np.array([[hessian[0, 0], hessian[0, 1], hessian[0, 2], gradient[0]],
                                            [hessian[1, 0], hessian[1, 1], hessian[1, 2], gradient[1]],
                                            [hessian[2, 0], hessian[2, 1], hessian[2, 2], gradient[2]],
                                            [gradient[0], gradient[1], gradient[2], 0.0]])
                    gradient_norm = gradient[0] ** 2 + gradient[1] ** 2 + gradient[2] ** 2

                    # Curvature computation
                    curvature = - np.linalg.det(ext_hessian) / (gradient_norm ** 2)

                    sorted_points.append((z, y, x, curvature))
                    curvatures[z, y, x] = curvature
            ProgressBar.update_progress_bar(debug, z / (size - 1))
        ProgressBar.end_progress_bar(debug)
        return curvatures, sorted_points

    def calculate_curvature_new(self, sdf, debug=True):
        """Compute gaussian curvature for each point in given array"""
        if debug:
            print('=> Computing numerical derivative and curvature of points...')
        sorted_points = []
        size = sdf.dimensions[0]
        curvatures = np.zeros((size, size, size), dtype=np.float32)
        ProgressBar.init_progress_bar(debug)
        for z in range(size):
            for y in range(size):
                for x in range(size):
                    # Disregard points that are not on the surface
                    if not sdf.on_surface(np.array((z, y, x)))[0]:
                        continue

                    # Curvature computation
                    # gradient = sdf.gradient(np.array((z, y, x)))
                    curvature = sdf.curvature(np.array((z, y, x)), self.epsilon)

                    # Option 1.1: determinant of cut 2x2 matrix
                    # hessian = curvature[0:2, 0:2]
                    # res = np.linalg.det(hessian)

                    # Option 1.2: eigenvalues of cut 2x2 matrix
                    # hessian = curvature[0:2, 0:2]
                    # eigen = np.linalg.eigvals(hessian)
                    # res = eigen[0] * eigen[1]

                    # Option 1.3: determinant of cut 2x2 matrix slightly changed
                    # hessian = curvature[0:2, 0:2]
                    # res = np.linalg.det(hessian) / (1 + gradient[0] ** 2 + gradient[1] ** 2) ** 2

                    # Option 2.1: eigenvalues of 3x3 matrix as curvature
                    eigen = np.linalg.eigvals(curvature)
                    res = eigen[0] * eigen[1] * eigen[2]

                    # Option 2.2: determinant of 3x3 matrix
                    # res = np.linalg.det(curvature)

                    # Option 2.3: eigenvalues of 3x3 matrix as mean curvature
                    # eigen = np.linalg.eigvals(curvature)
                    # res = (eigen[0] + eigen[1]) / 2

                    # Option 3: formula from above
                    # ext_hessian = np.array([[curvature[0, 0], curvature[0, 1], curvature[0, 2], gradient[0]],
                    #                         [curvature[1, 0], curvature[1, 1], curvature[1, 2], gradient[1]],
                    #                         [curvature[2, 0], curvature[2, 1], curvature[2, 2], gradient[2]],
                    #                         [gradient[0], gradient[1], gradient[2], 0.0]])
                    # gradient_norm = np.linalg.norm(gradient, 1)
                    # res = - np.linalg.det(ext_hessian) / (gradient_norm ** 4)

                    # Option 4: formula from iso-surface
                    # A = np.matrix(curvature)
                    # adj_hessian = A.getH()
                    # res = (gradient.reshape((1, 3)).dot(adj_hessian.dot(gradient.reshape(3, 1)))) /
                    #        np.linalg.norm(gradient, 2) ** 4

                    sorted_points.append((z, y, x, abs(res)))
                    curvatures[z, y, x] = abs(res)
            ProgressBar.update_progress_bar(debug, z / (size - 1))
        ProgressBar.end_progress_bar(debug)
        return curvatures, sorted_points

    def classify_points(self, points, sorted_points, debug=True):
        """Assign class to each point based on sorted curvature, meaning to find points of high curvature"""
        if debug:
            print('=> Sorting curvature of points...')
        sorted_points.sort(key=lambda elem: elem[3], reverse=False)
        size = np.shape(points)[0]
        points_of_interest = np.zeros((size, size, size), dtype=np.int32)
        percentile = np.percentile(sorted_points, self.percentage)
        print(percentile)

        # Assign positive class 1 to points of high curvature up to certain percentage
        # for i in range(0, int(len(sorted_points) * (self.percentage / 100.0))):
        for i in range(len(sorted_points)):
            if sorted_points[i][3] < percentile:
                continue
            z = sorted_points[i][0]
            y = sorted_points[i][1]
            x = sorted_points[i][2]
            points_of_interest[z, y, x] = 1
        return points_of_interest
