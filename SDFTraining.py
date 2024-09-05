import numpy as np
import torch.utils.data
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
        # Check if we have GPU available to run tensors on
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        # Available datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Fixed parameters
        self.min_epochs = 10
        self.max_epochs = 100
        self.early_exit_iterations = 3
        self.pred_folder = 'pred/'
        self.pred_file_extension = '.bin'
        self.prediction_num = 5
        self.model_type = model_type
        self.grid_search = grid_search

        # Parameters found during grid search (if enabled)
        self.learning_rate = 0.001
        self.batch_size = 8
        self.loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1])).to(self.device)

    def init_model(self):
        if self.model_type == 'unet2':
            return SDFUnetLevel2().to(self.device)
        elif self.model_type == 'unet3':
            return SDFUnetLevel3().to(self.device)
        else:
            raise ValueError(f'Unknown model type \"{self.model_type}\"')

    def print_training_parameters(self, log_file, learning_rate, batch_size, loss_function, model_description):
        log_str = f'=> Starting training...\n' \
                  f'   Learning rate: {learning_rate}\n' \
                  f'   Batch size:    {batch_size}\n' \
                  f'   Loss function: {loss_function.__class__.__name__}{vars(loss_function)}\n' \
                  f'{str(model_description)}\n'
        log_file.write(log_str)
        print(log_str)

    def print_epoch(self, log_file, epoch, metrics, loss):
        log_str = f'=> Epoch ({epoch + 1})\n' \
                  f'   => Validation set summary:\n' \
                  f'    - Accuracy:  {metrics.accuracy * 100:.2f}% ({metrics.accuracy})\n' \
                  f'    - Precision: {metrics.precision * 100:.2f}% ({metrics.precision})\n' \
                  f'    - Recall:    {metrics.recall * 100:.2f}% ({metrics.recall})\n' \
                  f'    - F1 score:  {metrics.f1_score * 100:.2f}% ({metrics.f1_score})\n' \
                  f'    - mIOU:      {metrics.mIOU * 100:.2f}% ({metrics.mIOU})\n' \
                  f'    - Loss:      {loss}\n'
        log_file.write(log_str)
        print(log_str)

    def train(self):
        if self.grid_search:
            # loss_functions = [nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.1])),
            #                   nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1])),
            #                   nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10])),
            #                   nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100])),
            #                   DiceLoss(),
            #                   TverskyLoss(0.1),
            #                   TverskyLoss(0.5),
            #                   TverskyLoss(0.9),
            #                   FocalTverskyLoss(0.1, 2),
            #                   FocalTverskyLoss(0.5, 2),
            #                   FocalTverskyLoss(0.9, 2)]
            # learning_rates = [0.1, 0.01, 0.001, 0.0001]
            # batch_sizes = [1, 2, 4, 8, 16, 32]
            loss_functions = [nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1]))]
            learning_rates = [0.001]
            batch_sizes = [4, 8]
            grid = itertools.product(loss_functions, learning_rates, batch_sizes)
            self.gridsearch(grid)

            # Train one last time on train + validation set, then evaluate on test set
            self.train_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.val_dataset])
            self.trainonce()
        else:
            self.trainonce()

    def gridsearch(self, grid):
        # Prepare datasets
        full_dataset = SDFDataset('samples\\', 'out\\', 1000)
        if self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
            split = [0.6, 0.2, 0.2]
            self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(full_dataset, split)

        best_mIOU = 0.0
        # TODO: add header to csv file
        for loss_function, learning_rate, batch_size in grid:
            train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)

            # Get dimension of input data (most likely always 64 * 64 * 64)
            train_features, _ = next(iter(train_dataloader))
            dim_x, dim_y, dim_z = train_features.size()[2], train_features.size()[3], train_features.size()[4]

            # Initialize everything needed for training and push to GPU if possible
            loss_function = loss_function.to(self.device)
            model = self.init_model()
            model_description = torchinfo.summary(model, (1, 1, dim_x, dim_y, dim_z), verbose=0)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            date = time.strftime('%Y%m%d%H%M')
            with (open(f'{self.pred_folder}{date}_validation_log.txt', 'w', encoding='utf-8') as log_file):
                self.print_training_parameters(log_file, learning_rate, batch_size, loss_function, model_description)

                metrics = Metrics(self.device)
                last_validation_loss = 0.0
                early_exit_count = 0
                current_epoch = 0

                for t in range(self.max_epochs):
                    current_epoch = t
                    metrics.reset_metrics()

                    # Training loop
                    model.train()
                    for batch, (X, y) in enumerate(train_dataloader):
                        # Compute prediction of current model and compute loss
                        prediction = model(X)
                        train_loss = loss_function(prediction, y)

                        # Do backpropagation
                        train_loss.backward()
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    # Validation loop
                    model.eval()
                    validation_loss = 0
                    sigmoid = nn.Sigmoid().to(self.device)
                    with torch.no_grad():
                        for X, y in self.val_dataset:
                            # Predict output of one validation sample
                            prediction = model(X.reshape((1, 1, dim_x, dim_y, dim_z)))
                            validation_loss += loss_function(prediction, y.unsqueeze(dim=0)).item()

                            # Update metrics (accuracy, precision, recall, f1) of test samples.
                            prediction = sigmoid(prediction)
                            prediction_conv = prediction.reshape((dim_x ** 3))
                            label_conv = y.reshape((dim_x ** 3)).int()
                            metrics.update(prediction_conv, label_conv)
                    metrics.compute()
                    validation_loss /= len(self.val_dataset)

                    self.print_epoch(log_file, t, metrics, validation_loss)
                    metrics.append()

                    # Check if validation loss was not improved over last few iterations = early exit
                    if t > self.min_epochs and validation_loss > last_validation_loss:
                        log_str = f'   => No improvement this epoch ({early_exit_count + 1} in row)\n'
                        print(log_str)
                        log_file.write(log_str)
                        early_exit_count += 1
                        last_validation_loss = validation_loss
                    else:
                        early_exit_count = 0
                        last_validation_loss = validation_loss
                    if early_exit_count >= self.early_exit_iterations:
                        log_str = f'   => Terminated due to early exit\n'
                        print(log_str)
                        log_file.write(log_str)
                        break

                # If current model outperforms current best model in grid search, then save parameters
                if metrics.mIOU > best_mIOU:
                    self.loss_function = loss_function
                    self.learning_rate = learning_rate
                    self.batch_size = batch_size
                    best_mIOU = metrics.mIOU

                    log_str = f'=> Found new best performing parameters (mIOU = {metrics.mIOU}):\n' \
                              f'   Learning rate: {self.learning_rate}\n' \
                              f'   Batch size:    {self.batch_size}\n' \
                              f'   Loss function: {self.loss_function.__class__.__name__}{vars(self.loss_function)}\n'
                    print(log_str)
                    log_file.write(log_str)

                with (open(f'{self.pred_folder}grid_search.csv', 'a', encoding='utf-8') as csv_file):
                    csv_file.write(f'{self.loss_function.__class__.__name__},{vars(self.loss_function)},'
                                   f'{learning_rate},{batch_size},{current_epoch},{metrics.accuracy},'
                                   f'{metrics.precision},{metrics.recall},{metrics.f1_score},{metrics.mIOU}\n')

                metrics.reset_metrics()
                metrics.reset_lists()

        log_str = f'=> Grid search done, best performing parameters:\n' \
                  f'   Learning rate: {self.learning_rate}\n' \
                  f'   Batch size:    {self.batch_size}\n' \
                  f'   Loss function: {self.loss_function.__class__.__name__}{vars(self.loss_function)}\n'
        print(log_str)
        # TODO: immediately try best model on test data set (save each best model)?

    def trainonce(self):
        # Prepare datasets and dataloaders
        full_dataset = SDFDataset('samples\\', 'out\\', 1000)
        if self.train_dataset is None or self.test_dataset is None:
            split = [0.8, 0.2]
            self.train_dataset, self.test_dataset = torch.utils.data.random_split(full_dataset, split)
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

        # Get dimension of input data (most likely always 64 * 64 * 64)
        train_features, _ = next(iter(train_dataloader))
        dim_x, dim_y, dim_z = train_features.size()[2], train_features.size()[3], train_features.size()[4]

        # Initialize everything needed for training and push to GPU if possible
        loss_function = self.loss_function.to(self.device)
        model = self.init_model()
        model_description = torchinfo.summary(model, (1, 1, dim_x, dim_y, dim_z), verbose=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        date = time.strftime('%Y%m%d%H%M')
        with (open(f'{self.pred_folder}{date}_test_log.txt', 'w', encoding='utf-8') as log_file):
            self.print_training_parameters(log_file, self.learning_rate, self.batch_size, loss_function,
                                           model_description)

            train_losses = []
            test_losses = []
            metrics = Metrics(self.device)
            last_test_loss = 0.0
            early_exit_count = 0

            for t in range(self.max_epochs):
                # Training loop
                model.train()
                train_loss = 0
                for batch, (X, y) in enumerate(train_dataloader):
                    # Compute prediction of current model and compute loss
                    prediction = model(X)
                    train_loss = loss_function(prediction, y)

                    # Do backpropagation
                    train_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                train_losses.append(train_loss.item())

                # Test loop
                model.eval()
                test_loss = 0
                sigmoid = nn.Sigmoid().to(self.device)
                with torch.no_grad():
                    for X, y in self.test_dataset:
                        # Predict output of one test sample
                        prediction = model(X.reshape((1, 1, dim_x, dim_y, dim_z)))
                        test_loss += loss_function(prediction, y.unsqueeze(dim=0)).item()

                        # Update metrics (accuracy, precision, recall, f1) of test samples
                        prediction = sigmoid(prediction)
                        prediction_conv = prediction.reshape((dim_x ** 3))
                        label_conv = y.reshape((dim_x ** 3)).int()
                        metrics.update(prediction_conv, label_conv)
                metrics.compute()
                test_loss /= len(self.test_dataset)

                self.print_epoch(log_file, t, metrics, test_loss)
                test_losses.append(test_loss)
                metrics.append()
                metrics.reset_metrics()

                # Check if test loss was not improved over last few iterations = early exit
                if t > self.min_epochs and test_loss > last_test_loss:
                    log_str = f'   => No improvement this epoch ({early_exit_count + 1} in row)\n'
                    print(log_str)
                    log_file.write(log_str)
                    early_exit_count += 1
                    last_test_loss = test_loss
                else:
                    early_exit_count = 0
                    last_test_loss = test_loss
                if early_exit_count >= self.early_exit_iterations:
                    log_str = f'   => Terminated due to early exit\n'
                    print(log_str)
                    log_file.write(log_str)
                    break

        # Save some predicted samples to pred/ folder (to visualize later)
        sigmoid = nn.Sigmoid()
        for i in range(self.prediction_num):
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
