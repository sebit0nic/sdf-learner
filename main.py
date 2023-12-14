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
        a = [[0] * 128] * 128
        plt.rcParams['axes.facecolor'] = 'blue'
        plt.xlim(0, 128)
        plt.ylim(0, 128)
        start_depth = 0
        file.seek(start_depth * 128 * 128 * 4)
        for z in range(32):
            for y in range(128):
                for x in range(128):
                    data = file.read(4)
                    dist = struct.unpack('f', data)[0]
                    if x % 2 == 0 and y % 2 == 0 and dist <= 0:
                        plt.plot(x, y, color='green', marker='o', markersize=3)
            plt.savefig('out/sdf-' + str(start_depth + z) + '.png')
            print('Done with z=' + str(start_depth + z))
        file.close()
