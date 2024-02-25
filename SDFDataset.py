from SDFFileHandler import SDFReader
import torch
import os
from torch.utils.data import Dataset


class SDFDataset(Dataset):
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder

    def __len__(self):
        files = os.listdir(os.getcwd() + '\\' + self.input_folder)
        return len(files)

    def __getitem__(self, item):
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'

        in_reader = SDFReader(self.input_folder + str(item + 1) + '.bin')
        in_samples = in_reader.read_samples_flat()
        in_tensor = torch.as_tensor(in_samples, dtype=torch.float32, device=device)

        out_reader = SDFReader(self.output_folder + str(item + 1) + '.csv')
        out_labels = out_reader.read_labels()
        out_tensor = torch.as_tensor(out_labels, dtype=torch.float32, device=device)

        return in_tensor, out_tensor
