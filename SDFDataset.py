from SDFFileHandler import SDFReader
import torch
from torch.utils.data import Dataset


class SDFDataset(Dataset):
    def __init__(self, input_folder, output_folder, sample_num):
        self.sample_num = sample_num

        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        in_reader = SDFReader(f'{input_folder}')
        self.in_tensor = in_reader.read_input_as_tensor(device, sample_num)
        out_reader = SDFReader(f'{output_folder}')
        self.out_tensor = out_reader.read_labels_as_tensor(device, sample_num)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, item):
        return self.in_tensor[item], self.out_tensor[item]
