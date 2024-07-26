from SDFVisualizer import ProgressBar
import os
import numpy as np
import torch


class SDFReader:
    def __init__(self, file_name):
        self.file_name = file_name

    @staticmethod
    def compute_dimensions_from_file(file):
        file.seek(0, os.SEEK_END)
        # BIN file: each distance / label is float32 / int32 => divide by 4
        size = np.round(np.power(file.tell() // 4, 1 / 3)).astype(int)
        file.seek(0, 0)
        return size

    def read_points_from_bin(self, is_label, debug=True):
        if debug:
            print('=> Reading in points from bin...')
            print('')
        file_path = os.getcwd() + '\\' + self.file_name
        file = open(file_path, 'rb')
        size = self.compute_dimensions_from_file(file)
        if is_label:
            points = np.fromfile(file, dtype=np.int32).reshape((size, size, size))
        else:
            points = np.fromfile(file, dtype=np.float32).reshape((size, size, size))
        file.close()
        return points

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
            data = np.fromfile(file, dtype=np.int32, sep=',')
            labels[i, 0] = np.copy(data).reshape((size, size, size))
            file.close()
            ProgressBar.update_progress_bar(debug, i / (sample_num - 1))
        ProgressBar.end_progress_bar(debug)
        print('')

        # Finally, convert array to tensor.
        return torch.as_tensor(labels, dtype=torch.float32, device=device)


class SDFWriter:
    def __init__(self, file_name):
        self.file_name = file_name

    def write_points(self, points_of_interest, debug=True):
        file_path = os.getcwd() + '\\' + self.file_name
        file = open(file_path, 'wb')
        if debug:
            print('=> Writing high estimated curvature points to file...\n')
        points_of_interest.tofile(file)
        file.close()
