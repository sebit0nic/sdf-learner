from SDFFileHandler import SDFReader
import torch
import os
from torch.utils.data import Dataset


class SDFDataset(Dataset):
    def __init__(self, input_folder, output_folder, sample_num):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.sample_num = sample_num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, item):
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'

        in_reader = SDFReader(f'{self.input_folder}sample{item:06d}_subdiv.bin')
        in_tensor = in_reader.read_input_as_tensor(device)

        out_reader = SDFReader(f'{self.output_folder}sample{item:06d}.csv')
        out_tensor = out_reader.read_labels_as_tensor(device)

        return in_tensor, out_tensor
