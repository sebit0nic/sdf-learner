from SDFVisualizer import ProgressBar
from Sample import Sample
import os
import struct


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
                    dist = struct.unpack('f', data)[0]
                    x_arr.append(Sample(dist))
                y_arr.append(x_arr)
            samples.append(y_arr)
            ProgressBar.update_progress_bar(z / 127)
        ProgressBar.end_progress_bar()
        print('')
        file.close()
        return samples


class SDFWriter:
    def __init__(self, file_name):
        self.file_name = file_name

    def write_samples(self, samples):
        file_path = os.getcwd() + '\\' + self.file_name
        file = open(file_path, 'wt')
        file.write('X,Y,Z\n')
        print('=> Writing high estimated curvature sample points to file...\n')
        for z in range(128):
            for y in range(128):
                for x in range(128):
                    if samples[z][y][x].high_curvature:
                        file.write(str(z) + ',' + str(y) + ',' + str(x) + '\n')
        file.close()
