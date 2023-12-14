import argparse
import os
import struct
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("========================================SDF Learner========================================================")
    parser = argparse.ArgumentParser(prog='SDF Learner',
                                     description='Learns the estimated curvature of a signed distance field.')
    parser.add_argument('-s', '--sdf_in', help='Binary file that contains a signed distance field learned manually.')
    args = parser.parse_args()
    matplotlib.use('TkAgg')
    if args.sdf_in is not None:
        file_path = os.getcwd() + '\\' + args.sdf_in
        print("Input SDF file: " + file_path)
        file = open(file_path, 'rb')
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.view_init(elev=30, azim=45, roll=0)
        X = []
        Y = []
        Z = []
        for z in range(0, 128):
            X.clear()
            Y.clear()
            Z.clear()
            for y in range(128):
                for x in range(128):
                    data = file.read(4)
                    dist = struct.unpack('f', data)[0]
                    if x % 8 == 0 and y % 8 == 0 and z % 8 == 0 and dist <= 0:
                        X.append(x)
                        Y.append(y)
                        Z.append(z)
            ax.scatter(X, Z, Y, color='green', marker='o')
            ax.set_xlim3d(0, 128)
            ax.set_ylim3d(0, 128)
            ax.set_zlim3d(0, 128)
            plt.savefig('out/sdf.png')
            print('Done with z=' + str(z))
        file.close()
