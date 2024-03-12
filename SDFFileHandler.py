from SDFVisualizer import ProgressBar
from SDFPoint import SDFPoint
import os
import struct
import csv
import numpy as np


class SDFReader:
    def __init__(self, file_name):
        self.file_name = file_name

    def read_configuration(self):
        print('=> Reading in configuration...')
        file_path = os.getcwd() + '\\' + self.file_name
        file = open(file_path, 'r')
        line = file.readline()
        args = line.split(';')
        return int(args[0]), float(args[1]), float(args[2])

    def read_points(self):
        print('=> Reading in points...')
        points = []
        file_path = os.getcwd() + '\\' + self.file_name
        file = open(file_path, 'rb')
        file.seek(0, os.SEEK_END)
        size = np.round(np.power(file.tell() // 4, 1 / 3)).astype(int)
        file.seek(0, 0)
        ProgressBar.init_progress_bar()
        for z in range(size):
            y_arr = []
            for y in range(size):
                x_arr = []
                for x in range(size):
                    data = file.read(4)
                    distance = struct.unpack('f', data)[0]
                    x_arr.append(SDFPoint(distance))
                y_arr.append(x_arr)
            points.append(y_arr)
            ProgressBar.update_progress_bar(z / (size - 1))
        ProgressBar.end_progress_bar()
        print('')
        file.close()
        return points, size

    def read_point_distances(self):
        points = np.zeros((128, 128, 128))
        file_path = os.getcwd() + '\\' + self.file_name
        file = open(file_path, 'rb')
        for z in range(128):
            for y in range(128):
                for x in range(128):
                    data = file.read(4)
                    distance = struct.unpack('f', data)[0]
                    points[z, y, x] = distance
        file.close()
        return points

    def read_labels(self):
        file_path = os.getcwd() + '\\' + self.file_name
        file = open(file_path, 'r')
        labels_flat = list(map(int, list(csv.reader(file))[0]))
        labels = np.zeros((128, 128, 128))
        i = 0
        for z in range(128):
            for y in range(128):
                for x in range(128):
                    labels[z, y, x] = labels_flat[i]
                    i += 0
        return labels


class SDFWriter:
    def __init__(self, file_name, size):
        self.file_name = file_name
        self.size = size

    def write_points(self, points):
        file_path = os.getcwd() + '\\' + self.file_name
        file = open(file_path, 'wt')
        print('=> Writing high estimated curvature points to file...\n')
        for z in range(self.size):
            for y in range(self.size):
                for x in range(self.size):
                    if x == 0 and y == 0 and z == 0:
                        file.write(str(points[z][y][x].high_curvature))
                    else:
                        file.write(',' + str(points[z][y][x].high_curvature))
        file.close()
