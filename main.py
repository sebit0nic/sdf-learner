import argparse
import math
import os
import struct
import numpy as np
import pyvista

if __name__ == "__main__":
    print("========================================SDF Learner========================================================")
    parser = argparse.ArgumentParser(prog='SDF Learner',
                                     description='Learns the estimated curvature of a signed distance field.')
    parser.add_argument('-i', '--sdf_in', help='Binary file that contains a signed distance field learned manually.')
    parser.add_argument('-d', '--density', help='Density of sampled points when SDF is exported.')
    parser.add_argument('-s', '--point_size', help='Point size of sampled points when SDF is exported.')
    parser.add_argument('-e', '--derivative', action='store', const='NoValue', nargs='?',
                        help='Calculate the derivative of the given SDF samples.')
    parser.add_argument('-m', '--matplot', action='store', const='NoValue', nargs='?',
                        help='Export graphics as matplotlib graph.')
    parser.add_argument('-p', '--pyvista', help='Export graphics as pyvista graph.')
    args = parser.parse_args()
    density = 1
    point_size = 5
    if args.density is not None:
        density = int(args.density)
    if args.point_size is not None:
        point_size = float(args.point_size)

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
            print('Done with z=' + str(z))
        file.close()

    epsilon = 0.1
    curv = []
    if args.derivative:
        print('=> Computing numerical derivative of samples...')
        for z in range(128):
            for y in range(128):
                for x in range(128):
                    # Central difference of x
                    x_neg = samples[z][y][x] if x - 1 < 0 else samples[z][y][x - 1]
                    x_neg_i = samples[z][y][x] + epsilon * (x_neg - samples[z][y][x])
                    x_pos = samples[z][y][x] if x + 1 >= 128 else samples[z][y][x + 1]
                    x_pos_i = samples[z][y][x] + epsilon * (x_pos - samples[z][y][x])
                    # Central difference of y
                    y_neg = samples[z][y][x] if y - 1 < 0 else samples[z][y - 1][x]
                    y_neg_i = samples[z][y][x] + epsilon * (y_neg - samples[z][y][x])
                    y_pos = samples[z][y][x] if y + 1 >= 128 else samples[z][y + 1][x]
                    y_pos_i = samples[z][y][x] + epsilon * (y_pos - samples[z][y][x])
                    # Central difference of z
                    z_neg = samples[z][y][x] if z - 1 < 0 else samples[z - 1][y][x]
                    z_neg_i = samples[z][y][x] + epsilon * (z_neg - samples[z][y][x])
                    z_pos = samples[z][y][x] if z + 1 >= 128 else samples[z + 1][y][x]
                    z_pos_i = samples[z][y][x] + epsilon * (z_pos - samples[z][y][x])
                    # First order derivative
                    x_dx = (x_pos_i - x_neg_i) / (2 * epsilon)
                    y_dy = (y_pos_i - y_neg_i) / (2 * epsilon)
                    z_dz = (z_pos_i - z_neg_i) / (2 * epsilon)
                    # Second order derivative
                    x_dx2 = (x_pos_i - (2 * samples[z][y][x]) + x_neg_i) / (epsilon ** 2)
                    y_dy2 = (y_pos_i - (2 * samples[z][y][x]) + y_neg_i) / (epsilon ** 2)
                    z_dz2 = (z_pos_i - (2 * samples[z][y][x]) + z_neg_i) / (epsilon ** 2)
                    curvature = (math.sqrt((z_dz2 * y_dy - y_dy2 * z_dz) ** 2 + (x_dx2 * z_dz - z_dz2 * x_dx) ** 2 +
                                           (y_dy2 * x_dx - x_dx2 * y_dy) ** 2)) / (math.sqrt(x_dx ** 2 + y_dy ** 2 +
                                                                                   z_dz ** 2)) ** 3
                    curv.append((x, y, z, curvature))
            print('Done with z=' + str(z))

    if args.pyvista is not None:
        arr_out = []
        arr_in = []
        arr_curv = []
        print('=> Visualizing samples using pyvista...')
        curv.sort(key=lambda elem: elem[3], reverse=True)
        target_points = int((float(args.pyvista) / 100.0) * (128 ** 3))
        for i in range(len(curv)):
            x = curv[i][0]
            y = curv[i][1]
            z = curv[i][2]
            if i < target_points:
                print(curv[i])
                arr_curv.append((float(x), float(y), float(z)))
            elif samples[z][y][x] <= 0:
                arr_in.append((float(x), float(y), float(z)))
            else:
                arr_out.append((float(x), float(y), float(z)))
        plotter = pyvista.Plotter()
        # if len(arr_out) != 0:
        #     pc_out = np.array(arr_out)
        #     plotter.add_mesh(pc_out, color='red', point_size=point_size, render_points_as_spheres=True, opacity=0.005)
        if len(arr_in) != 0:
            pc_in = np.array(arr_in)
            plotter.add_mesh(pc_in, color='green', point_size=point_size, render_points_as_spheres=True, opacity=1)
        if len(arr_curv) != 0:
            pc_curv = np.array(arr_curv)
            plotter.add_mesh(pc_curv, color='blue', point_size=point_size, render_points_as_spheres=True, opacity=1)
        plotter.show_axes()
        plotter.show_grid()
        plotter.export_obj('out/sdf.obj')
        plotter.show()
