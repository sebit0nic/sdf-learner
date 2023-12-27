import argparse
import math
import os
import struct
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyvista

if __name__ == "__main__":
    print("========================================SDF Learner========================================================")
    parser = argparse.ArgumentParser(prog='SDF Learner',
                                     description='Learns the estimated curvature of a signed distance field.')
    parser.add_argument('-s', '--sdf_in', help='Binary file that contains a signed distance field learned manually.')
    parser.add_argument('-d', '--density', help='Density of sampled points when SDF is exported.')
    parser.add_argument('-p', '--point_size', help='Point size of sampled points when SDF is exported.')
    parser.add_argument('-e', '--derivative', action='store', const='NoValue', nargs='?',
                        help='Calculate the derivative of the given SDF samples.')
    parser.add_argument('-m', '--matplot', action='store', const='NoValue', nargs='?',
                        help='Export graphics as matplotlib graph.')
    parser.add_argument('-y', '--pyvista', help='Export graphics as pyvista graph.')
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
        # TODO: https://en.wikipedia.org/wiki/Curvature#Graph_of_a_function
        for z in range(128):
            for y in range(128):
                for x in range(128):
                    x_l = samples[z][y][x] if x - 1 < 0 else samples[z][y][x - 1]
                    x_li = (x_l - samples[z][y][x]) / epsilon
                    x_r = samples[z][y][x] if x + 1 >= 128 else samples[z][y][x + 1]
                    x_ri = (x_r - samples[z][y][x]) / epsilon
                    y_u = samples[z][y][x] if y - 1 < 0 else samples[z][y - 1][x]
                    y_ui = (y_u - samples[z][y][x]) / epsilon
                    y_d = samples[z][y][x] if y + 1 >= 128 else samples[z][y + 1][x]
                    y_di = (y_d - samples[z][y][x]) / epsilon
                    z_f = samples[z][y][x] if z - 1 < 0 else samples[z - 1][y][x]
                    z_fi = (z_f - samples[z][y][x]) / epsilon
                    z_b = samples[z][y][x] if z + 1 >= 128 else samples[z + 1][y][x]
                    z_bi = (z_b - samples[z][y][x]) / epsilon
                    x_dx = (x_ri - (2 * samples[z][y][x]) + x_li) / (epsilon ** 2)
                    y_dy = (y_di - (2 * samples[z][y][x]) + y_ui) / (epsilon ** 2)
                    z_dz = (z_bi - (2 * samples[z][y][x]) + z_fi) / (epsilon ** 2)
                    mag = math.sqrt((x_dx ** 2) + (y_dy ** 2) + (z_dz ** 2))
                    curv.append((x, y, z, mag))
            print('Done with z=' + str(z))

    if args.matplot:
        matplotlib.use('TkAgg')
        X = []
        Y = []
        Z = []
        print('=> Visualizing samples using matplotlib...')
        for z in range(128):
            for y in range(128):
                for x in range(128):
                    if x % density == 0 and y % density == 0 and z % density == 0:
                        if samples[z][y][x] <= 0:
                            X.append(x)
                            Y.append(y)
                            Z.append(z)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(projection='3d'))
        for i in range(3):
            ax[i].view_init(elev=30, azim=i * 45, roll=0)
            ax[i].set_xlabel('x')
            ax[i].set_ylabel('y')
            ax[i].set_zlabel('z')
            ax[i].scatter(X, Y, Z, color='green', marker='o')
            ax[i].set_xlim3d(0, 128)
            ax[i].set_ylim3d(0, 128)
            ax[i].set_zlim3d(0, 128)
        plt.savefig('out/sdf.png')

    if args.pyvista is not None:
        arr_out = []
        arr_in = []
        arr_curv = []
        print('=> Visualizing samples using pyvista...')
        curv.sort(key=lambda elem: elem[3])
        target_points = int((float(args.pyvista) / 100.0) * (128 ** 3))
        for i in range(len(curv)):
            x = curv[i][0]
            y = curv[i][1]
            z = curv[i][2]
            if i < target_points:
                arr_curv.append((x * 0.1, y * 0.1, z * 0.1))
            elif samples[z][y][x] <= 0:
                arr_in.append((x * 0.1, y * 0.1, z * 0.1))
            else:
                arr_out.append((x * 0.1, y * 0.1, z * 0.1))
        plotter = pyvista.Plotter()
        if len(arr_out) != 0:
            pc_out = np.array(arr_out)
            plotter.add_mesh(pc_out, color='red', point_size=point_size, render_points_as_spheres=True, opacity=0.005)
        if len(arr_in) != 0:
            pc_in = np.array(arr_in)
            plotter.add_mesh(pc_in, color='green', point_size=point_size, render_points_as_spheres=True, opacity=1)
        if len(arr_curv) != 0:
            pc_curv = np.array(arr_curv)
            plotter.add_mesh(pc_curv, color='blue', point_size=point_size, render_points_as_spheres=True, opacity=1)
        plotter.show_axes()
        plotter.show()
