"""
File name: SDFFileHandler.py
Author: Sebastian Lackner
Version: 1.0
Description: Handling of all file I/O related tasks
"""

import os
import numpy as np
import torch

from SDFVisualizer import ProgressBar


class SDFReader:
    """Object for reading tasks on file"""

    def __init__(self, file_name):
        self.file_name = file_name

    @staticmethod
    def compute_dimensions_from_file(file):
        """Compute number of points in each dimension using the file size"""
        file.seek(0, os.SEEK_END)
        # BIN file: each distance / label is float32 / int32 => divide by 4
        size = np.round(np.power(file.tell() // 4, 1 / 3)).astype(int)
        file.seek(0, 0)
        return size

    def read_points_from_bin(self, is_label, debug=True):
        """Read either class labels or point data from file"""
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

    def read_dataset_from_bin(self, device, sample_num, is_label, debug=True):
        """Deprecated"""
        # Initialize array by using first available sample.
        if is_label:
            file_path = f'{os.getcwd()}\\{self.file_name}sample000000.bin'
        else:
            file_path = f'{os.getcwd()}\\{self.file_name}sample000000_subdiv.bin'
        file = open(file_path, 'rb')
        size = self.compute_dimensions_from_file(file)
        points = np.zeros((sample_num, 1, size, size, size))
        file.close()

        # Loop over all samples and store them in array.
        if is_label:
            print(f'=> Init dataset labels...')
        else:
            print(f'=> Init dataset samples...')
        ProgressBar.init_progress_bar(debug)
        for i in range(sample_num):
            if is_label:
                file_path = f'{os.getcwd()}\\{self.file_name}sample{i:06d}.bin'
            else:
                file_path = f'{os.getcwd()}\\{self.file_name}sample{i:06d}_subdiv.bin'
            file = open(file_path, 'rb')
            if is_label:
                data = np.fromfile(file, dtype=np.int32)
            else:
                data = np.fromfile(file, dtype=np.float32)
            points[i, 0] = np.copy(data).reshape((size, size, size))
            file.close()
            ProgressBar.update_progress_bar(debug, i / (sample_num - 1))
        ProgressBar.end_progress_bar(debug)
        print('')

        # Finally, convert array to tensor.
        return torch.as_tensor(points, dtype=torch.float32, device=device)

    def read_sample_from_bin(self, device, item, is_label):
        """Read either labels or distances from file for training dataset"""
        if is_label:
            file_path = f'{os.getcwd()}\\{self.file_name}sample{item:06d}.bin'
        else:
            file_path = f'{os.getcwd()}\\{self.file_name}sample{item:06d}_subdiv.bin'
        file = open(file_path, 'rb')
        size = self.compute_dimensions_from_file(file)
        points = np.zeros((1, size, size, size))
        if is_label:
            data = np.fromfile(file, dtype=np.int32)
        else:
            data = np.fromfile(file, dtype=np.float32)
        points[0] = np.copy(data).reshape((size, size, size))
        file.close()
        return torch.as_tensor(points, dtype=torch.float32, device=device)


class SDFWriter:
    """Object for writing tasks on file"""

    def __init__(self, file_name):
        self.file_name = file_name

    def write_points(self, points_of_interest, debug=True):
        """Write ground truth or prediction class labels to file"""
        file_path = os.getcwd() + '\\' + self.file_name
        file = open(file_path, 'wb')
        if debug:
            print('=> Writing high estimated curvature points to file...\n')
        points_of_interest.tofile(file)
        file.close()
