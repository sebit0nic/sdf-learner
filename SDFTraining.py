"""
File name: SDFTraining.py
Author: Sebastian Lackner
Version: 1.0
Description: Handling of machine learning training
"""

import numpy as np
import torch.utils.data
import torch.amp
import torchinfo
import itertools
import time
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from SDFDataset import SDFDataset
from SDFMetrics import Metrics
from SDFFileHandler import SDFWriter


class DiceLoss(nn.Module):
    """Implementation of dice loss function used during training"""

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets):
        logits = self.sigmoid(inputs)
        TP = (logits * targets).sum(dim=(2, 3, 4))
        total = (targets + logits).sum(dim=(2, 3, 4))
        return (1 - ((2 * TP + 1) / (total + 1))).mean()


class TverskyLoss(nn.Module):
    """Implementation of tversky loss function used during training"""

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
    """Implementation of focal tversky loss function used during training"""

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


class SDFUnetLevel2(nn.Module):
    """Implementation of 2-level-deep U-Net architecture"""

    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv12 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv21 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv22 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv31 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv32 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv33 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv41 = nn.Conv3d(in_channels=192, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv42 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv51 = nn.Conv3d(in_channels=96, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv52 = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.deconv1 = nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                          padding=(1, 1, 1), output_padding=(1, 1, 1))
        self.deconv2 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                          padding=(1, 1, 1), output_padding=(1, 1, 1))
        self.ReLU = nn.ReLU()
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

    def forward(self, x):
        logits = self.conv11(x)
        logits = self.ReLU(logits)
        logits = self.conv12(logits)
        logits = self.ReLU(logits)
        skip1 = torch.clone(logits)
        logits = self.max_pool(logits)
        logits = self.conv21(logits)
        logits = self.ReLU(logits)
        logits = self.conv22(logits)
        logits = self.ReLU(logits)
        skip2 = torch.clone(logits)
        logits = self.max_pool(logits)
        logits = self.conv31(logits)
        logits = self.ReLU(logits)
        logits = self.conv32(logits)
        logits = self.ReLU(logits)
        logits = self.conv33(logits)
        logits = self.ReLU(logits)
        logits = self.deconv1(logits)
        logits = torch.cat((logits, skip2), dim=1)
        logits = self.conv41(logits)
        logits = self.ReLU(logits)
        logits = self.conv42(logits)
        logits = self.ReLU(logits)
        logits = self.deconv2(logits)
        logits = torch.cat((logits, skip1), dim=1)
        logits = self.conv51(logits)
        logits = self.ReLU(logits)
        logits = self.conv52(logits)
        return logits


class SDFUnetLevel3(nn.Module):
    """Implementation of 3-level-deep U-Net architecture"""

    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv12 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv21 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv22 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv31 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv32 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv41 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv42 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv51 = nn.Conv3d(in_channels=768, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv52 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv61 = nn.Conv3d(in_channels=384, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv62 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv71 = nn.Conv3d(in_channels=192, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv72 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv73 = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.deconv1 = nn.ConvTranspose3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                          padding=(1, 1, 1), output_padding=(1, 1, 1))
        self.deconv2 = nn.ConvTranspose3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                          padding=(1, 1, 1), output_padding=(1, 1, 1))
        self.deconv3 = nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                          padding=(1, 1, 1), output_padding=(1, 1, 1))
        self.ReLU = nn.ReLU()
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

    def forward(self, x):
        logits = self.conv11(x)
        logits = self.ReLU(logits)
        logits = self.conv12(logits)
        logits = self.ReLU(logits)
        skip1 = torch.clone(logits)
        logits = self.max_pool(logits)
        logits = self.conv21(logits)
        logits = self.ReLU(logits)
        logits = self.conv22(logits)
        logits = self.ReLU(logits)
        skip2 = torch.clone(logits)
        logits = self.max_pool(logits)
        logits = self.conv31(logits)
        logits = self.ReLU(logits)
        logits = self.conv32(logits)
        logits = self.ReLU(logits)
        skip3 = torch.clone(logits)
        logits = self.max_pool(logits)
        logits = self.conv41(logits)
        logits = self.ReLU(logits)
        logits = self.conv42(logits)
        logits = self.ReLU(logits)
        logits = self.deconv1(logits)
        logits = torch.cat((logits, skip3), dim=1)
        logits = self.conv51(logits)
        logits = self.ReLU(logits)
        logits = self.conv52(logits)
        logits = self.ReLU(logits)
        logits = self.deconv2(logits)
        logits = torch.cat((logits, skip2), dim=1)
        logits = self.conv61(logits)
        logits = self.ReLU(logits)
        logits = self.conv62(logits)
        logits = self.ReLU(logits)
        logits = self.deconv3(logits)
        logits = torch.cat((logits, skip1), dim=1)
        logits = self.conv71(logits)
        logits = self.ReLU(logits)
        logits = self.conv72(logits)
        logits = self.ReLU(logits)
        logits = self.conv73(logits)
        return logits


class SDFSegnet(nn.Module):
    """Implementation of SegNet architecture (experimental)"""

    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv12 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv21 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv22 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv31 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv32 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv41 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv42 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv51 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv52 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv61 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv62 = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.ReLU = nn.ReLU()
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), return_indices=True)
        self.max_unpool = nn.MaxUnpool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

    def forward(self, x):
        logits = self.conv11(x)
        logits = self.ReLU(logits)
        logits = self.conv12(logits)
        logits = self.ReLU(logits)
        size1 = logits.size()
        logits, indices1 = self.max_pool(logits)
        logits = self.conv21(logits)
        logits = self.ReLU(logits)
        logits = self.conv22(logits)
        logits = self.ReLU(logits)
        size2 = logits.size()
        logits, indices2 = self.max_pool(logits)
        logits = self.conv31(logits)
        logits = self.ReLU(logits)
        logits = self.conv32(logits)
        logits = self.ReLU(logits)
        size3 = logits.size()
        logits, indices3 = self.max_pool(logits)
        logits = self.max_unpool(logits, indices3, size3)
        logits = self.conv41(logits)
        logits = self.ReLU(logits)
        logits = self.conv42(logits)
        logits = self.ReLU(logits)
        logits = self.max_unpool(logits, indices2, size2)
        logits = self.conv51(logits)
        logits = self.ReLU(logits)
        logits = self.conv52(logits)
        logits = self.ReLU(logits)
        logits = self.max_unpool(logits, indices1, size1)
        logits = self.conv61(logits)
        logits = self.ReLU(logits)
        logits = self.conv62(logits)
        return logits


class SDFTrainer:
    """Object to handle machine learning training tasks"""

    def __init__(self, model_type, grid_search, in_folder, out_folder):
        # Check if we have GPU available to run tensors on
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        # Available datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Fixed parameters
        self.min_epochs = 20
        self.max_epochs = 100
        self.early_exit_iterations = 3
        self.pred_folder = 'pred/'
        self.pred_file_extension = '.bin'
        self.prediction_num = 5
        self.model_type = model_type
        self.grid_search = grid_search
        self.in_folder = in_folder
        self.out_folder = out_folder

        # Parameters found during grid search (if enabled)
        self.learning_rate = 0.001
        self.batch_size = 2
        self.loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1])).to(self.device)

    def init_model(self):
        """Initialize model architecture used for training based on given token"""
        if self.model_type == 'unet2':
            return SDFUnetLevel2().to(self.device)
        elif self.model_type == 'unet3':
            return SDFUnetLevel3().to(self.device)
        elif self.model_type == 'seg':
            return SDFSegnet().to(self.device)
        else:
            raise ValueError(f'Unknown model type \"{self.model_type}\"')

    def print_training_parameters(self, log_file, learning_rate, batch_size, loss_function, model_description):
        """Print hyperparameters used during training"""
        log_str = f'=> Starting training...\n' \
                  f'   Learning rate: {learning_rate}\n' \
                  f'   Batch size:    {batch_size}\n' \
                  f'   Loss function: {loss_function.__class__.__name__}{vars(loss_function)}\n' \
                  f'{str(model_description)}\n'
        log_file.write(log_str)
        print(log_str)

    def print_epoch(self, log_file, epoch, metrics, loss):
        """Print statistics computed during one training epoch"""
        log_str = f'=> Epoch ({epoch + 1})\n' \
                  f'   => Validation/test set summary:\n' \
                  f'    - Accuracy:  {metrics.accuracy * 100:.2f}% ({metrics.accuracy})\n' \
                  f'    - Precision: {metrics.precision * 100:.2f}% ({metrics.precision})\n' \
                  f'    - Recall:    {metrics.recall * 100:.2f}% ({metrics.recall})\n' \
                  f'    - F1 score:  {metrics.f1_score * 100:.2f}% ({metrics.f1_score})\n' \
                  f'    - mIOU:      {metrics.mIOU * 100:.2f}% ({metrics.mIOU})\n' \
                  f'    - Loss:      {loss}\n'
        log_file.write(log_str)
        print(log_str)

    def train(self):
        """Train the model either once or using grid search on some selected hyperparameters"""
        if self.grid_search:
            loss_functions = [nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1])),
                              nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.1])),
                              nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10])),
                              DiceLoss(),
                              FocalTverskyLoss(0.1, 2),
                              FocalTverskyLoss(0.5, 2),
                              FocalTverskyLoss(0.9, 2)]
            learning_rates = [0.001, 0.0001]
            batch_sizes = [8, 4, 2, 1]
            grid = itertools.product(loss_functions, learning_rates, batch_sizes)
            self.gridsearch(grid)

            # Train one last time on train + validation set, then evaluate on test set
            self.train_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.val_dataset])
            self.trainonce()
        else:
            self.trainonce()

    def gridsearch(self, grid):
        """Do grid search run yielding best performing hyperparameter combination"""
        # Prepare datasets
        full_dataset = SDFDataset(self.in_folder, self.out_folder, 1000)
        if self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
            split = [0.6, 0.2, 0.2]
            self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(full_dataset, split)

        best_mIOU = 0.0
        with (open(f'{self.pred_folder}grid_search.csv', 'a', encoding='utf-8') as csv_file):
            csv_file.write(f'loss_function;parameters;learning_rate;batch_size;epochs;accuracy;precision;recall;'
                           f'f1_score;m_iou\n')
        for loss_function, learning_rate, batch_size in grid:
            train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

            # Get dimension of input data (most likely always 64 * 64 * 64)
            train_features, _ = next(iter(train_dataloader))
            dim_x, dim_y, dim_z = train_features.size()[2], train_features.size()[3], train_features.size()[4]

            # Initialize everything needed for training and push to GPU if possible
            loss_function = loss_function.to(self.device)
            model = self.init_model()
            model_description = torchinfo.summary(model, (1, 1, dim_x, dim_y, dim_z), verbose=0)
            optimizer = torch.optim.Adam(model.parameters(), eps=1e-4, lr=learning_rate)
            torch.set_float32_matmul_precision('medium')
            torch.backends.cudnn.benchmark = True
            scaler = torch.amp.GradScaler()

            date = time.strftime('%Y%m%d%H%M')
            with (open(f'{self.pred_folder}{date}_validation_log.txt', 'a', encoding='utf-8') as log_file):
                self.print_training_parameters(log_file, learning_rate, batch_size, loss_function, model_description)

            metrics = Metrics(self.device)
            last_validation_loss = 0.0
            early_exit_count = 0
            current_epoch = 0
            current_mIOU = 0.0

            for t in range(self.max_epochs):
                current_epoch = t
                metrics.reset_metrics()

                # Training loop
                model.train()
                for batch, (X, y) in enumerate(train_dataloader):
                    optimizer.zero_grad(set_to_none=True)

                    # Compute prediction of current model and compute loss
                    with torch.autocast(device_type=self.device):
                        prediction = model(X)
                        train_loss = loss_function(prediction, y)

                    # Do backpropagation
                    scaler.scale(train_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                # Validation loop
                model.eval()
                validation_loss = torch.zeros(1, device=self.device)
                sigmoid = nn.Sigmoid().to(self.device)
                with torch.no_grad():
                    for X, y in self.val_dataset:
                        # Predict output of one validation sample
                        prediction = model(X.reshape((1, 1, dim_x, dim_y, dim_z)))
                        validation_loss += loss_function(prediction, y.unsqueeze(dim=0))

                        # Update metrics (accuracy, precision, recall, f1) of test samples.
                        prediction = sigmoid(prediction)
                        prediction_conv = prediction.reshape((dim_x ** 3))
                        label_conv = y.reshape((dim_x ** 3)).int()
                        metrics.update(prediction_conv, label_conv)
                metrics.compute()
                validation_loss /= len(self.val_dataset)

                with (open(f'{self.pred_folder}{date}_validation_log.txt', 'a', encoding='utf-8') as log_file):
                    self.print_epoch(log_file, t, metrics, validation_loss.item())
                metrics.append()

                # Check if validation loss was not improved over last few iterations = early exit
                if t >= self.min_epochs and (validation_loss >= last_validation_loss or torch.isnan(validation_loss)):
                    log_str = f'   => No improvement this epoch ({early_exit_count + 1} in row)\n'
                    print(log_str)
                    with (open(f'{self.pred_folder}{date}_validation_log.txt', 'a', encoding='utf-8') as log_file):
                        log_file.write(log_str)
                    early_exit_count += 1
                    last_validation_loss = validation_loss
                else:
                    early_exit_count = 0
                    last_validation_loss = validation_loss
                    current_mIOU = metrics.mIOU
                if early_exit_count >= self.early_exit_iterations:
                    log_str = f'   => Terminated due to early exit\n'
                    print(log_str)
                    with (open(f'{self.pred_folder}{date}_validation_log.txt', 'a', encoding='utf-8') as log_file):
                        log_file.write(log_str)
                    break

            # If current model outperforms current best model in grid search, then save parameters
            if current_mIOU > best_mIOU:
                self.loss_function = loss_function
                self.learning_rate = learning_rate
                self.batch_size = batch_size
                best_mIOU = current_mIOU

                log_str = f'=> Found new best performing parameters (mIOU = {current_mIOU}):\n' \
                          f'   Learning rate: {self.learning_rate}\n' \
                          f'   Batch size:    {self.batch_size}\n' \
                          f'   Loss function: {self.loss_function.__class__.__name__}{vars(self.loss_function)}\n'
                print(log_str)
                with (open(f'{self.pred_folder}{date}_validation_log.txt', 'a', encoding='utf-8') as log_file):
                    log_file.write(log_str)

            with (open(f'{self.pred_folder}grid_search.csv', 'a', encoding='utf-8') as csv_file):
                csv_file.write(f'{loss_function.__class__.__name__};{vars(loss_function)};'
                               f'{learning_rate};{batch_size};{current_epoch + 1};{metrics.accuracy};'
                               f'{metrics.precision};{metrics.recall};{metrics.f1_score};{metrics.mIOU}\n')

            metrics.reset_metrics()
            metrics.reset_lists()

        log_str = f'=> Grid search done, best performing parameters:\n' \
                  f'   Learning rate: {self.learning_rate}\n' \
                  f'   Batch size:    {self.batch_size}\n' \
                  f'   Loss function: {self.loss_function.__class__.__name__}{vars(self.loss_function)}\n'
        print(log_str)

    def trainonce(self):
        """Do training run yielding test performance and predictions"""
        # Prepare datasets and dataloaders
        full_dataset = SDFDataset(self.in_folder, self.out_folder, 1000)
        if self.train_dataset is None or self.test_dataset is None:
            split = [0.8, 0.2]
            self.train_dataset, self.test_dataset = torch.utils.data.random_split(full_dataset, split)
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        # Get dimension of input data (most likely always 64 * 64 * 64)
        train_features, _ = next(iter(train_dataloader))
        dim_x, dim_y, dim_z = train_features.size()[2], train_features.size()[3], train_features.size()[4]

        # Initialize everything needed for training and push to GPU if possible
        loss_function = self.loss_function.to(self.device)
        model = self.init_model()
        model_description = torchinfo.summary(model, (1, 1, dim_x, dim_y, dim_z), verbose=0)
        optimizer = torch.optim.Adam(model.parameters(), eps=1e-4, lr=self.learning_rate)
        torch.set_float32_matmul_precision('medium')
        torch.backends.cudnn.benchmark = True
        scaler = torch.amp.GradScaler()

        date = time.strftime('%Y%m%d%H%M')
        with (open(f'{self.pred_folder}{date}_test_log.txt', 'a', encoding='utf-8') as log_file):
            self.print_training_parameters(log_file, self.learning_rate, self.batch_size, loss_function,
                                           model_description)

        train_losses = []
        test_losses = []
        metrics = Metrics(self.device)
        last_test_loss = 0.0
        early_exit_count = 0
        model_params = model.state_dict()

        for t in range(self.max_epochs):
            # Training loop
            model.train()
            train_loss = 0
            for batch, (X, y) in enumerate(train_dataloader):
                optimizer.zero_grad(set_to_none=True)

                # Compute prediction of current model and compute loss
                with torch.autocast(device_type=self.device):
                    prediction = model(X)
                    train_loss = loss_function(prediction, y)

                # Do backpropagation
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            train_losses.append(train_loss.item())

            # Test loop
            model.eval()
            test_loss = torch.zeros(1, device=self.device)
            sigmoid = nn.Sigmoid().to(self.device)
            with torch.no_grad():
                for X, y in self.test_dataset:
                    # Predict output of one test sample
                    prediction = model(X.reshape((1, 1, dim_x, dim_y, dim_z)))
                    test_loss += loss_function(prediction, y.unsqueeze(dim=0))

                    # Update metrics (accuracy, precision, recall, f1) of test samples
                    prediction = sigmoid(prediction)
                    prediction_conv = prediction.reshape((dim_x ** 3))
                    label_conv = y.reshape((dim_x ** 3)).int()
                    metrics.update(prediction_conv, label_conv)
            metrics.compute()
            test_loss /= len(self.test_dataset)

            with (open(f'{self.pred_folder}{date}_test_log.txt', 'a', encoding='utf-8') as log_file):
                self.print_epoch(log_file, t, metrics, test_loss.item())
            test_losses.append(test_loss.item())
            metrics.append()
            metrics.reset_metrics()

            # Check if test loss was not improved over last few iterations = early exit
            if t >= self.min_epochs and (test_loss >= last_test_loss or torch.isnan(test_loss)):
                log_str = f'   => No improvement this epoch ({early_exit_count + 1} in row)\n'
                print(log_str)
                with (open(f'{self.pred_folder}{date}_test_log.txt', 'a', encoding='utf-8') as log_file):
                    log_file.write(log_str)
                early_exit_count += 1
                last_test_loss = test_loss
            else:
                early_exit_count = 0
                last_test_loss = test_loss
                model_params = model.state_dict()
            if early_exit_count >= self.early_exit_iterations:
                log_str = f'   => Terminated due to early exit\n'
                print(log_str)
                with (open(f'{self.pred_folder}{date}_test_log.txt', 'a', encoding='utf-8') as log_file):
                    log_file.write(log_str)
                break

        # Save some predicted samples to pred/ folder (to visualize later)
        sigmoid = nn.Sigmoid()
        for i in range(self.prediction_num):
            # Load best model first in case there was an early exit
            model.load_state_dict(model_params)
            model.eval()

            o_path = f'{self.pred_folder}{date}_{str(i).zfill(6)}{self.pred_file_extension}'
            prediction = sigmoid(model(full_dataset[i][0].reshape((1, 1, 64, 64, 64))))
            sdf_writer = SDFWriter(o_path)
            prediction_conv = prediction.squeeze().round().numpy(force=True).astype(np.int32)
            sdf_writer.write_points(prediction_conv)

        plt.plot(train_losses, color='blue')
        plt.plot(test_losses, color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over epochs')
        plt.savefig(f'{self.pred_folder}{date}_loss.png')
        plt.show()

        metrics.plot(self.pred_folder, date)

        train_losses.clear()
        test_losses.clear()
        metrics.reset_metrics()
        metrics.reset_lists()
