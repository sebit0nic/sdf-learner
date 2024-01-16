from SDFFileHandler import SDFReader
from SDFFileHandler import SDFWriter
import argparse
import numpy as np
import pyvista
import time


if __name__ == "__main__":
    print("========================================SDF Learner========================================================")
    parser = argparse.ArgumentParser(prog='SDF Learner',
                                     description='Learns the curvature of a signed distance field.')
    parser.add_argument('-i', '--sdf_in', help='Binary file that contains a signed distance field.')
    parser.add_argument('-o', '--sdf_out', help='File where estimated points of high curvature are written to.')
    parser.add_argument('-s', '--point_size', help='Point size of sampled points when SDF is exported.')
    parser.add_argument('-t', '--tolerance', help='Tolerance of the SDF distance values.')
    parser.add_argument('-r', '--percentage', help='Percentage of points considered for curvature estimation.')
    parser.add_argument('-c', '--curvature', action='store', const='Set', nargs='?',
                        help='Calculate the derivative and curvature of the given SDF samples.')
    parser.add_argument('-p', '--pyvista', action='store', const='Set', nargs='?',
                        help='Export graphics as pyvista graph.')
    args = parser.parse_args()

    point_size = 5
    tolerance = 0.1
    percentage = 1.0
    if args.point_size is not None:
        point_size = float(args.point_size)
    if args.tolerance is not None:
        tolerance = float(args.tolerance)
    if args.percentage is not None:
        percentage = float(args.percentage)
    print('=> Parameters:')
    print('   SDF input file:      ' + str(args.sdf_in))
    print('   SDF output file:     ' + str(args.sdf_out))
    print('   Point size:          ' + str(point_size))
    print('   Tolerance:           ' + str(tolerance))
    print('   Percentage:          ' + str(percentage))
    print('   Calculate curvature: ' + str(args.curvature == 'Set'))
    print('   Show graph:          ' + str(args.pyvista == 'Set'))
    time.sleep(2)
    print('')

    samples = []
    if args.sdf_in is not None:
        sdf_reader = SDFReader(args.sdf_in)
        samples = sdf_reader.read_samples()

    if args.curvature:
        print('=> Computing numerical derivative and curvature of samples...')
        sorted_samples = []
        target_points = 0
        epsilon = 0.1
        minima = 0
        maxima = 0
        print('   Progress: ' + 100 * '.', end='')
        for z in range(128):
            for y in range(128):
                for x in range(128):
                    # Disregard points on the borders
                    if x - 1 < 0 or x + 1 >= 128 or y - 1 < 0 or y + 1 >= 128 or z - 1 < 0 or z + 1 >= 128:
                        sorted_samples.append((z, y, x, 0.0))
                        samples[z][y][x].curvature = 0.0
                        continue
                    # Disregard points being nowhere near the surface
                    if samples[z][y][x].distance < -tolerance or samples[z][y][x].distance > tolerance:
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
            print('\r   Progress: ' + (int((z / 127) * 100) * '#') + (100 - int((z / 127) * 100)) * '.',
                  end='', flush=True)
        print('')
        print('   Minimum curvature found: ' + str(minima))
        print('   Maximum curvature found: ' + str(maxima))
        print('')

        print('=> Sorting curvature of samples...\n')
        sorted_samples.sort(key=lambda elem: abs(elem[3]), reverse=True)
        target_points = int((float(percentage) / 100.0) * (128 ** 3))
        for i in range(target_points):
            z = sorted_samples[i][0]
            y = sorted_samples[i][1]
            x = sorted_samples[i][2]
            samples[z][y][x].high_curvature = True

    if args.sdf_out is not None:
        sdf_writer = SDFWriter(args.sdf_out)
        sdf_writer.write_samples(samples)

    if args.pyvista:
        arr_in = []
        arr_curv_pos = []
        arr_curv_neg = []
        cur_points = 0
        print('=> Visualizing sample points using pyvista...')
        print('   Progress: ' + 100 * '.', end='')
        for z in range(128):
            for y in range(128):
                for x in range(128):
                    if samples[z][y][x].high_curvature:
                        if samples[z][y][x].curvature > 0:
                            arr_curv_pos.append((float(x), float(y), float(z)))
                        else:
                            arr_curv_neg.append((float(x), float(y), float(z)))
                    elif samples[z][y][x].distance <= 0:
                        arr_in.append((float(x), float(y), float(z)))
            print('\r   Progress: ' + (int((z / 127) * 100) * '#') + (100 - int((z / 127) * 100)) * '.',
                  end='', flush=True)
        print('')
        plotter = pyvista.Plotter()
        plotter.add_mesh(pyvista.Box(bounds=(0.0, 128.0, 0.0, 128.0, 0.0, 128.0)), color='red', opacity=0.01)
        if len(arr_in) != 0:
            pc_in = np.array(arr_in)
            plotter.add_mesh(pc_in, color='green', point_size=point_size, render_points_as_spheres=True, opacity=1)
        if len(arr_curv_pos) != 0:
            pc_curv = np.array(arr_curv_pos)
            plotter.add_mesh(pc_curv, color='yellow', point_size=point_size, render_points_as_spheres=True, opacity=1)
        if len(arr_curv_neg) != 0:
            pc_curv = np.array(arr_curv_neg)
            plotter.add_mesh(pc_curv, color='blue', point_size=point_size, render_points_as_spheres=True, opacity=1)
        plotter.show_axes()
        plotter.show_grid()
        plotter.export_obj('out/sdf.obj')
        plotter.show()
