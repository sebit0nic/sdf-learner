from SDFVisualizer import ProgressBar
from SDFPoint import SDFPoint
import os
import struct
import csv
import numpy as np
import torch


class SDFReader:
    def __init__(self, file_name):
        self.file_name = file_name

    @staticmethod
    def compute_dimensions_from_file(file, is_label):
        file.seek(0, os.SEEK_END)
        if is_label:
            # CSV file: each label is inbetween comma => divide by 2
            size = np.round(np.power(file.tell() // 2, 1 / 3)).astype(int)
        else:
            # BIN file: each distance is float => divide by 4
            size = np.round(np.power(file.tell() // 4, 1 / 3)).astype(int)
        file.seek(0, 0)
        return size

    def read_points_from_bin(self, debug=True):
        # TODO: use numpy arrays
        if debug:
            print('=> Reading in points from bin...')
        points = []
        file_path = os.getcwd() + '\\' + self.file_name
        file = open(file_path, 'rb')
        size = self.compute_dimensions_from_file(file, False)
        ProgressBar.init_progress_bar(debug)
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
            ProgressBar.update_progress_bar(debug, z / (size - 1))
        ProgressBar.end_progress_bar(debug)
        print('')
        file.close()
        return points, size

    def read_points_from_csv(self, debug=True):
        # TODO: use numpy arrays
        if debug:
            print('=> Reading in points from csv...')
        points = []
        file_path = os.getcwd() + '\\' + self.file_name
        file = open(file_path, 'r')
        size = self.compute_dimensions_from_file(file, True)
        labels_flat = list(map(int, list(csv.reader(file))[0]))
        i = 0
        ProgressBar.init_progress_bar(debug)
        for z in range(size):
            y_arr = []
            for y in range(size):
                x_arr = []
                for x in range(size):
                    p = SDFPoint(0)
                    p.high_curvature = labels_flat[i]
                    x_arr.append(p)
                    i += 1
                y_arr.append(x_arr)
            points.append(y_arr)
            ProgressBar.update_progress_bar(debug, z / (size - 1))
        ProgressBar.end_progress_bar(debug)
        print('')
        file.close()
        return points, size

    def read_input_as_tensor(self, device, sample_num, debug=True):
        # Initialize array by using first available sample.
        file_path = f'{os.getcwd()}\\{self.file_name}sample000000_subdiv.bin'
        file = open(file_path, 'rb')
        size = self.compute_dimensions_from_file(file, False)
        points = np.zeros((sample_num, 1, size, size, size))
        file.close()

        # Loop over all samples and store them in array.
        print(f'=> Init dataset samples...')
        ProgressBar.init_progress_bar(debug)
        for i in range(sample_num):
            file_path = f'{os.getcwd()}\\{self.file_name}sample{i:06d}_subdiv.bin'
            file = open(file_path, 'rb')
            data = np.fromfile(file, dtype=np.float32)
            points[i, 0] = np.copy(data).reshape((size, size, size))
            file.close()
            ProgressBar.update_progress_bar(debug, i / (sample_num - 1))
        ProgressBar.end_progress_bar(debug)
        print('')

        # Finally, convert array to tensor.
        return torch.as_tensor(points, dtype=torch.float32, device=device)

    def read_labels_as_tensor(self, device, sample_num, debug=True):
        # Initialize array by using first available label.
        file_path = f'{os.getcwd()}\\{self.file_name}sample000000.csv'
        file = open(file_path, 'r')
        size = self.compute_dimensions_from_file(file, True)
        labels = np.zeros((sample_num, 1, size, size, size))
        file.close()

        # Loop over all labels and store them in array.
        print(f'=> Init dataset labels...')
        ProgressBar.init_progress_bar(debug)
        for i in range(sample_num):
            file_path = f'{os.getcwd()}\\{self.file_name}sample{i:06d}.csv'
            file = open(file_path, 'r')
            labels_flat = list(map(int, list(csv.reader(file))[0]))
            j = 0
            # TODO: read in data once, then split into numpy array
            for z in range(size):
                for y in range(size):
                    for x in range(size):
                        labels[i, 0, z, y, x] = labels_flat[j]
                        j += 1
            file.close()
            ProgressBar.update_progress_bar(debug, i / (sample_num - 1))
        ProgressBar.end_progress_bar(debug)
        print('')

        # Finally, convert array to tensor.
        return torch.as_tensor(labels, dtype=torch.float32, device=device)


class SDFWriter:
    def __init__(self, file_name, size):
        self.file_name = file_name
        self.size = size

    def write_points(self, points, debug=True):
        file_path = os.getcwd() + '\\' + self.file_name
        file = open(file_path, 'wt')
        if debug:
            print('=> Writing high estimated curvature points to file...\n')
        for z in range(self.size):
            for y in range(self.size):
                for x in range(self.size):
                    if x == 0 and y == 0 and z == 0:
                        file.write(str(points[z][y][x].high_curvature))
                    else:
                        file.write(',' + str(points[z][y][x].high_curvature))
        file.close()
