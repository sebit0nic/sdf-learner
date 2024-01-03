import argparse
import os
import struct
import numpy as np
import pyvista

if __name__ == "__main__":
    print("========================================SDF Learner========================================================")
    parser = argparse.ArgumentParser(prog='SDF Learner',
                                     description='Learns the estimated curvature of a signed distance field.')
    parser.add_argument('-i', '--sdf_in', help='Binary file that contains a signed distance field learned manually.')
    parser.add_argument('-s', '--point_size', help='Point size of sampled points when SDF is exported.')
    parser.add_argument('-t', '--tolerance', help='Tolerance of the SDF distance values.')
    parser.add_argument('-c', '--curvature', action='store', const='NoValue', nargs='?',
                        help='Calculate the derivative and curvature of the given SDF samples.')
    parser.add_argument('-p', '--pyvista', help='Export graphics as pyvista graph.')
    args = parser.parse_args()
    point_size = 5
    tolerance = 0.1
    if args.point_size is not None:
        point_size = float(args.point_size)
    if args.tolerance is not None:
        tolerance = float(args.tolerance)

    samples = []
    print('=> Reading in samples...')
    if args.sdf_in is not None:
        file_path = os.getcwd() + '\\' + args.sdf_in
        print("Input SDF file: " + file_path)
        file = open(file_path, 'rb')
        for z in range(128):
            y_arr = []
            for y in range(128):
                x_arr = []
                for x in range(128):
                    data = file.read(4)
                    dist = struct.unpack('f', data)[0]
                    x_arr.append(dist)
                y_arr.append(x_arr)
            samples.append(y_arr)
            print('Done with reading in samples at layer z=' + str(z))
        file.close()

    epsilon = 0.1
    curv = []
    minima = 0
    maxima = 0
    if args.curvature:
        print('=> Computing numerical derivative and curvature of samples...')
        for z in range(128):
            for y in range(128):
                for x in range(128):
                    # TODO: handle this better
                    if x - 1 < 0 or x + 1 >= 128 or y - 1 < 0 or y + 1 >= 128 or z - 1 < 0 or z + 1 >= 128:
                        curv.append((x, y, z, 0))
                        continue
                    # Interpolate x
                    x_neg_i = samples[z][y][x] + epsilon * (samples[z][y][x - 1] - samples[z][y][x])
                    x_pos_i = samples[z][y][x] + epsilon * (samples[z][y][x + 1] - samples[z][y][x])
                    xy_pos_i = samples[z][y][x] + epsilon * (samples[z][y + 1][x + 1] - samples[z][y][x])
                    xy_neg_i = samples[z][y][x] + epsilon * (samples[z][y - 1][x - 1] - samples[z][y][x])
                    # Interpolate y
                    y_neg_i = samples[z][y][x] + epsilon * (samples[z][y - 1][x] - samples[z][y][x])
                    y_pos_i = samples[z][y][x] + epsilon * (samples[z][y + 1][x] - samples[z][y][x])
                    xz_pos_i = samples[z][y][x] + epsilon * (samples[z + 1][y][x + 1] - samples[z][y][x])
                    xz_neg_i = samples[z][y][x] + epsilon * (samples[z - 1][y][x - 1] - samples[z][y][x])
                    # Interpolate z
                    z_neg_i = samples[z][y][x] + epsilon * (samples[z - 1][y][x] - samples[z][y][x])
                    z_pos_i = samples[z][y][x] + epsilon * (samples[z + 1][y][x] - samples[z][y][x])
                    yz_pos_i = samples[z][y][x] + epsilon * (samples[z + 1][y + 1][x] - samples[z][y][x])
                    yz_neg_i = samples[z][y][x] + epsilon * (samples[z - 1][y - 1][x] - samples[z][y][x])
                    # Second order derivative
                    f_dx2 = (x_pos_i - (2 * samples[z][y][x]) + x_neg_i) / (epsilon ** 2)
                    f_dy2 = (y_pos_i - (2 * samples[z][y][x]) + y_neg_i) / (epsilon ** 2)
                    f_dz2 = (z_pos_i - (2 * samples[z][y][x]) + z_neg_i) / (epsilon ** 2)
                    f_dxy = ((xy_pos_i - x_pos_i - y_pos_i + 2 * samples[z][y][x] - x_neg_i - y_neg_i + xy_neg_i) /
                             2 * (epsilon ** 2))
                    f_dxz = ((xz_pos_i - x_pos_i - z_pos_i + 2 * samples[z][y][x] - x_neg_i - z_neg_i + xz_neg_i) /
                             2 * (epsilon ** 2))
                    f_dyz = ((yz_pos_i - y_pos_i - z_pos_i + 2 * samples[z][y][x] - y_neg_i - z_neg_i + yz_neg_i) /
                             2 * (epsilon ** 2))
                    # Curvature computation
                    curvature = (f_dx2 * (f_dy2 * f_dz2 - f_dyz * f_dxz) - f_dxy * (f_dxy * f_dz2 - f_dyz * f_dxz) +
                                 f_dxz * (f_dxy * f_dyz - f_dy2 * f_dxz))
                    if curvature < minima:
                        minima = curvature
                    if curvature > maxima:
                        maxima = curvature
                    curv.append((x, y, z, curvature))
            print('Done with derivative at layer z=' + str(z))
        print('Minimum curvature found: ' + str(minima))
        print('Maximum curvature found: ' + str(maxima))
    else:
        for z in range(128):
            for y in range(128):
                for x in range(128):
                    curv.append((x, y, z, 0))

    if args.pyvista is not None:
        arr_in = []
        arr_curv_pos = []
        arr_curv_neg = []
        print('=> Visualizing samples using pyvista...')
        target_points = 0
        if args.curvature:
            curv.sort(key=lambda elem: abs(elem[3]), reverse=True)
            target_points = int((float(args.pyvista) / 100.0) * (128 ** 3))
        cur_points = 0
        for i in range(len(curv)):
            x = curv[i][0]
            y = curv[i][1]
            z = curv[i][2]
            if cur_points < target_points and -tolerance <= samples[z][y][x] <= 0:
                if curv[i][3] < 0:
                    arr_curv_neg.append((float(x), float(y), float(z)))
                else:
                    arr_curv_pos.append((float(x), float(y), float(z)))
                cur_points += 1
            elif samples[z][y][x] <= 0:
                arr_in.append((float(x), float(y), float(z)))
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
