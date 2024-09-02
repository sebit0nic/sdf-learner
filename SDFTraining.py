import torch.utils.data
from torch import nn
from torch.utils.data import DataLoader
import itertools
from SDFDataset import SDFDataset


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets):
        logits = self.sigmoid(inputs)
        TP = (logits * targets).sum(dim=(2, 3, 4))
        total = (targets + logits).sum(dim=(2, 3, 4))
        return (1 - ((2 * TP + 1) / (total + 1))).mean()


class TverskyLoss(nn.Module):
    def __init__(self, beta=0.5):
        super(TverskyLoss, self).__init__()
        self.beta = beta
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets):
        logits = self.sigmoid(inputs)
        TP = (logits * targets).sum(dim=(2, 3, 4))
        FN = (targets * (1 - logits)).sum(dim=(2, 3, 4))
        FP = ((1 - targets) * logits).sum(dim=(2, 3, 4))
        return (1 - (TP / (TP + (1 - self.beta) * FN + self.beta * FP))).mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, beta=0.5, gamma=2):
        super(FocalTverskyLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets):
        logits = self.sigmoid(inputs)
        TP = (logits * targets).sum(dim=(2, 3, 4))
        FN = (targets * (1 - logits)).sum(dim=(2, 3, 4))
        FP = ((1 - targets) * logits).sum(dim=(2, 3, 4))
        TI = TP / (TP + (1 - self.beta) * FN + self.beta * FP)
        return torch.pow(1 - TI, self.gamma).mean()


# TODO: implement SegNet
# TODO: implement FCN
class SDFUnetLevel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv5 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv6 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv7 = nn.Conv3d(in_channels=96, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv8 = nn.Conv3d(in_channels=48, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv9 = nn.Conv3d(in_channels=16, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.deconv1 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                          padding=(1, 1, 1), output_padding=(1, 1, 1))
        self.deconv2 = nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                          padding=(1, 1, 1), output_padding=(1, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

    def forward(self, x):
        logits = self.conv1(x)
        logits = self.ReLU(logits)
        logits = self.conv2(logits)
        logits = self.ReLU(logits)
        skip1 = torch.clone(logits)
        logits = self.max_pool(logits)
        logits = self.conv3(logits)
        logits = self.ReLU(logits)
        logits = self.conv4(logits)
        logits = self.ReLU(logits)
        skip2 = torch.clone(logits)
        logits = self.max_pool(logits)
        logits = self.conv5(logits)
        logits = self.ReLU(logits)
        logits = self.conv6(logits)
        logits = self.ReLU(logits)
        logits = self.conv6(logits)
        logits = self.ReLU(logits)
        logits = self.deconv1(logits)
        logits = torch.cat((logits, skip2), dim=1)
        logits = self.conv7(logits)
        logits = self.ReLU(logits)
        logits = self.deconv2(logits)
        logits = torch.cat((logits, skip1), dim=1)
        logits = self.conv8(logits)
        logits = self.ReLU(logits)
        logits = self.conv9(logits)
        return logits


class SDFUnetLevel3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv5 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv6 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv7 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv8 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv9 = nn.Conv3d(in_channels=192, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv10 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv11 = nn.Conv3d(in_channels=96, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv12 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv13 = nn.Conv3d(in_channels=48, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv14 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv15 = nn.Conv3d(in_channels=16, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.deconv1 = nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                          padding=(1, 1, 1), output_padding=(1, 1, 1))
        self.deconv2 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                          padding=(1, 1, 1), output_padding=(1, 1, 1))
        self.deconv3 = nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                          padding=(1, 1, 1), output_padding=(1, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

    def forward(self, x):
        logits = self.conv1(x)
        logits = self.ReLU(logits)
        logits = self.conv2(logits)
        logits = self.ReLU(logits)
        skip1 = torch.clone(logits)
        logits = self.max_pool(logits)
        logits = self.conv3(logits)
        logits = self.ReLU(logits)
        logits = self.conv4(logits)
        logits = self.ReLU(logits)
        skip2 = torch.clone(logits)
        logits = self.max_pool(logits)
        logits = self.conv5(logits)
        logits = self.ReLU(logits)
        logits = self.conv6(logits)
        logits = self.ReLU(logits)
        skip3 = torch.clone(logits)
        logits = self.max_pool(logits)
        logits = self.conv7(logits)
        logits = self.ReLU(logits)
        logits = self.conv8(logits)
        logits = self.ReLU(logits)
        logits = self.deconv1(logits)
        logits = torch.cat((logits, skip3), dim=1)
        logits = self.conv9(logits)
        logits = self.ReLU(logits)
        logits = self.conv10(logits)
        logits = self.ReLU(logits)
        logits = self.deconv2(logits)
        logits = torch.cat((logits, skip2), dim=1)
        logits = self.conv11(logits)
        logits = self.ReLU(logits)
        logits = self.conv12(logits)
        logits = self.ReLU(logits)
        logits = self.deconv3(logits)
        logits = torch.cat((logits, skip1), dim=1)
        logits = self.conv13(logits)
        logits = self.ReLU(logits)
        logits = self.conv14(logits)
        logits = self.ReLU(logits)
        logits = self.conv15(logits)
        return logits


class SDFTrainer:
    def __init__(self, model_type, grid_search):
        # Check if we have GPU available to run tensors on.
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        # Fixed parameters.
        self.min_epochs = 10
        self.max_epochs = 100
        self.grid_search = grid_search

        # Parameters found during grid search (if enabled).
        self.learning_rate = 0.001
        self.batch_size = 8
        self.loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1])).to(self.device)

    def train(self):
        loss_functions = [nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.1])),
                          nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1])),
                          nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10])),
                          nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100])),
                          DiceLoss(),
                          TverskyLoss(0.1),
                          TverskyLoss(0.5),
                          TverskyLoss(0.9),
                          FocalTverskyLoss(0.1, 2),
                          FocalTverskyLoss(0.5, 2),
                          FocalTverskyLoss(0.9, 2)]
        learning_rates = [1, 0.1, 0.01, 0.001, 0.0001]
        batch_sizes = [1, 2, 4, 8, 16, 32]
        grid = itertools.product(loss_functions, learning_rates, batch_sizes)

        full_dataset = SDFDataset('samples\\', 'out\\', 1000)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.6, 0.2, 0.2])

        # Initialize train + validation + test data loader with given batch size.
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Get dimension of input data (most likely always 64 * 64 * 64).
        train_features, _ = next(iter(train_dataloader))
        dim_x, dim_y, dim_z = train_features.size()[2], train_features.size()[3], train_features.size()[4]
