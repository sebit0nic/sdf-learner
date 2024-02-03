from SDFVisualizer import ProgressBar


class SDFCurvature:
    def __init__(self, epsilon, tolerance, percentage):
        self.epsilon = epsilon
        self.tolerance = tolerance
        self.percentage = percentage

    def calculate_curvature(self, samples):
        print('=> Computing numerical derivative and curvature of samples...')
        sorted_samples = []
        epsilon = 0.1
        minima = 0
        maxima = 0
        ProgressBar.init_progress_bar()
        for z in range(128):
            for y in range(128):
                for x in range(128):
                    # Disregard points on the borders
                    if x - 1 < 0 or x + 1 >= 128 or y - 1 < 0 or y + 1 >= 128 or z - 1 < 0 or z + 1 >= 128:
                        sorted_samples.append((z, y, x, 0.0))
                        samples[z][y][x].curvature = 0.0
                        continue
                    # Disregard points being nowhere near the surface
                    if samples[z][y][x].distance < -self.tolerance or samples[z][y][x].distance > self.tolerance:
                        sorted_samples.append((z, y, x, 0.0))
                        samples[z][y][x].curvature = 0.0
                        continue
                    # Interpolate x
                    x_neg_i = samples[z][y][x].distance + epsilon * (samples[z][y][x - 1].distance -
                                                                     samples[z][y][x].distance)
                    x_pos_i = samples[z][y][x].distance + epsilon * (samples[z][y][x + 1].distance -
                                                                     samples[z][y][x].distance)
                    xy_pos_i = samples[z][y][x].distance + epsilon * (samples[z][y + 1][x + 1].distance -
                                                                      samples[z][y][x].distance)
                    xy_neg_i = samples[z][y][x].distance + epsilon * (samples[z][y - 1][x - 1].distance -
                                                                      samples[z][y][x].distance)
                    # Interpolate y
                    y_neg_i = samples[z][y][x].distance + epsilon * (samples[z][y - 1][x].distance -
                                                                     samples[z][y][x].distance)
                    y_pos_i = samples[z][y][x].distance + epsilon * (samples[z][y + 1][x].distance -
                                                                     samples[z][y][x].distance)
                    yz_pos_i = samples[z][y][x].distance + epsilon * (samples[z + 1][y + 1][x].distance -
                                                                      samples[z][y][x].distance)
                    yz_neg_i = samples[z][y][x].distance + epsilon * (samples[z - 1][y - 1][x].distance -
                                                                      samples[z][y][x].distance)
                    # Interpolate z
                    z_neg_i = samples[z][y][x].distance + epsilon * (samples[z - 1][y][x].distance -
                                                                     samples[z][y][x].distance)
                    z_pos_i = samples[z][y][x].distance + epsilon * (samples[z + 1][y][x].distance -
                                                                     samples[z][y][x].distance)
                    xz_pos_i = samples[z][y][x].distance + epsilon * (samples[z + 1][y][x + 1].distance -
                                                                      samples[z][y][x].distance)
                    xz_neg_i = samples[z][y][x].distance + epsilon * (samples[z - 1][y][x - 1].distance -
                                                                      samples[z][y][x].distance)
                    # Second order derivative
                    f_dx2 = (x_pos_i - (2 * samples[z][y][x].distance) + x_neg_i) / (epsilon ** 2)
                    f_dy2 = (y_pos_i - (2 * samples[z][y][x].distance) + y_neg_i) / (epsilon ** 2)
                    f_dz2 = (z_pos_i - (2 * samples[z][y][x].distance) + z_neg_i) / (epsilon ** 2)
                    f_dxy = ((xy_pos_i - x_pos_i - y_pos_i + 2 * samples[z][y][x].distance - x_neg_i - y_neg_i +
                              xy_neg_i) / 2 * (epsilon ** 2))
                    f_dxz = ((xz_pos_i - x_pos_i - z_pos_i + 2 * samples[z][y][x].distance - x_neg_i - z_neg_i +
                              xz_neg_i) / 2 * (epsilon ** 2))
                    f_dyz = ((yz_pos_i - y_pos_i - z_pos_i + 2 * samples[z][y][x].distance - y_neg_i - z_neg_i +
                              yz_neg_i) / 2 * (epsilon ** 2))
                    # Curvature computation
                    curvature = (f_dx2 * (f_dy2 * f_dz2 - f_dyz * f_dxz) - f_dxy * (f_dxy * f_dz2 - f_dyz * f_dxz) +
                                 f_dxz * (f_dxy * f_dyz - f_dy2 * f_dxz))
                    if curvature < minima:
                        minima = curvature
                    if curvature > maxima:
                        maxima = curvature
                    sorted_samples.append((z, y, x, curvature))
                    samples[z][y][x].curvature = curvature
            ProgressBar.update_progress_bar(z / 127)
        ProgressBar.end_progress_bar()
        print('   Minimum curvature found: ' + str(minima))
        print('   Maximum curvature found: ' + str(maxima))
        print('')
        return samples, sorted_samples

    def classify_samples(self, samples, sorted_samples):
        print('=> Sorting curvature of samples...\n')
        sorted_samples.sort(key=lambda elem: abs(elem[3]), reverse=True)
        target_points = int((float(self.percentage) / 100.0) * (128 ** 3))
        for i in range(target_points):
            z = sorted_samples[i][0]
            y = sorted_samples[i][1]
            x = sorted_samples[i][2]
            samples[z][y][x].high_curvature = 1
        return samples
