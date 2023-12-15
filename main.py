import argparse
import os
import struct
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
        fig, ax = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(projection='3d'))
        ax[0].view_init(elev=30, azim=0, roll=0)
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        ax[0].set_zlabel('z')
        ax[1].view_init(elev=30, azim=45, roll=0)
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('y')
        ax[1].set_zlabel('z')
        ax[2].view_init(elev=30, azim=90, roll=0)
        ax[2].set_xlabel('x')
        ax[2].set_ylabel('y')
        ax[2].set_zlabel('z')
        X = []
        Y = []
        Z = []
        density = 8
        for z in range(0, 128):
            X.clear()
            Y.clear()
            Z.clear()
            for y in range(128):
                for x in range(128):
                    data = file.read(4)
                    dist = struct.unpack('f', data)[0]
                    if x % density == 0 and y % density == 0 and z % density == 0 and dist <= 0:
                        X.append(x)
                        Y.append(y)
                        Z.append(z)
            ax[0].scatter(X, Y, Z, color='green', marker='o')
            ax[0].set_xlim3d(0, 128)
            ax[0].set_ylim3d(0, 128)
            ax[0].set_zlim3d(0, 128)
            ax[1].scatter(X, Y, Z, color='green', marker='o')
            ax[1].set_xlim3d(0, 128)
            ax[1].set_ylim3d(0, 128)
            ax[1].set_zlim3d(0, 128)
            ax[2].scatter(X, Y, Z, color='green', marker='o')
            ax[2].set_xlim3d(0, 128)
            ax[2].set_ylim3d(0, 128)
            ax[2].set_zlim3d(0, 128)
            print('Done with z=' + str(z))
        plt.savefig('out/sdf.png')
        file.close()
