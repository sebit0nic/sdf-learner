from SDFVisualizer import ProgressBar
from SDFSample import SDFSample
import os
import struct
import csv


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

    def read_samples(self):
        print('=> Reading in samples...')
        samples = []
        file_path = os.getcwd() + '\\' + self.file_name
        file = open(file_path, 'rb')
        ProgressBar.init_progress_bar()
        for z in range(128):
            y_arr = []
            for y in range(128):
                x_arr = []
                for x in range(128):
                    data = file.read(4)
                    distance = struct.unpack('f', data)[0]
                    x_arr.append(SDFSample(distance))
                y_arr.append(x_arr)
            samples.append(y_arr)
            ProgressBar.update_progress_bar(z / 127)
        ProgressBar.end_progress_bar()
        print('')
        file.close()
        return samples

    def read_samples_flat(self):
        samples = []
        file_path = os.getcwd() + '\\' + self.file_name
        file = open(file_path, 'rb')
        for z in range(128):
            for y in range(128):
                for x in range(128):
                    data = file.read(4)
                    distance = struct.unpack('f', data)[0]
                    samples.append(distance)
        file.close()
        return samples

    def read_labels(self):
        file_path = os.getcwd() + '\\' + self.file_name
        file = open(file_path, 'r')
        labels = list(map(int, list(csv.reader(file))[0]))
        return labels


class SDFWriter:
    def __init__(self, file_name):
        self.file_name = file_name

    def write_samples(self, samples):
        file_path = os.getcwd() + '\\' + self.file_name
        file = open(file_path, 'wt')
        print('=> Writing high estimated curvature sample points to file...\n')
        for z in range(128):
            for y in range(128):
                for x in range(128):
                    if x == 0 and y == 0 and z == 0:
                        file.write(str(samples[z][y][x].high_curvature))
                    else:
                        file.write(',' + str(samples[z][y][x].high_curvature))
        file.close()
