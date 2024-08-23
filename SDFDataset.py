from SDFFileHandler import SDFReader
import torch
from torch.utils.data import Dataset


class SDFDataset(Dataset):
    def __init__(self, input_folder, output_folder, sample_num):
        self.sample_num = sample_num
        self.input_folder = input_folder
        self.output_folder = output_folder

    def __len__(self):
        return self.sample_num

    def __getitem__(self, item):
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'

        in_reader = SDFReader(f'{self.input_folder}')
        in_tensor = in_reader.read_sample_from_bin(device, item, False)
        out_reader = SDFReader(f'{self.output_folder}')
        out_tensor = out_reader.read_sample_from_bin(device, item, True)

        return in_tensor, out_tensor
