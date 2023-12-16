import argparse
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
    parser.add_argument('-m', '--matplot', action='store', const='NoValue', nargs='?',
                        help='Export graphics as matplotlib graph.')
    parser.add_argument('-y', '--pyvista', action='store', const='NoValue', nargs='?',
                        help='Export graphics as pyvista graph.')
    args = parser.parse_args()
    density = 1
    point_size = 5
    if args.density is not None:
        density = int(args.density)
    if args.point_size is not None:
        point_size = float(args.point_size)

    matplotlib.use('TkAgg')
    arr = []
    arr_s = []
    X = []
    Y = []
    Z = []
    if args.sdf_in is not None:
        file_path = os.getcwd() + '\\' + args.sdf_in
        print("Input SDF file: " + file_path)
        file = open(file_path, 'rb')
        for z in range(0, 128):
            for y in range(128):
                for x in range(128):
                    data = file.read(4)
                    dist = struct.unpack('f', data)[0]
                    if x % density == 0 and y % density == 0 and z % density == 0:
                        if dist <= 0:
                            arr.append((float(x * 0.1), float(y * 0.1), float(z * 0.1)))
                            arr_s.append((0, 1, 0, 1))
                            X.append(x)
                            Y.append(y)
                            Z.append(z)
                        else:
                            arr.append((float(x * 0.1), float(y * 0.1), float(z * 0.1)))
                            arr_s.append((1, 0, 0, 0.01))
            print('Done with z=' + str(z))
        file.close()

    if args.matplot:
        point_cloud = np.array(arr)
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
    if args.pyvista:
        point_cloud = np.array(arr)
        rgba = np.array(arr_s)
        pyvista.plot(point_cloud, scalars=rgba, render_points_as_spheres=True, point_size=point_size,
                     show_scalar_bar=False, rgba=True)
