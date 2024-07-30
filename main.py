import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader
from torch import nn
from torcheval.metrics import BinaryAccuracy
from torcheval.metrics import BinaryPrecision
from torcheval.metrics import BinaryRecall
from torcheval.metrics import BinaryF1Score

from SDFFileHandler import SDFReader
from SDFFileHandler import SDFWriter
from SDFUtil import SDFCurvature
from SDFVisualizer import SDFVisualizer
from SDFDataset import SDFDataset
import argparse
import time
import matplotlib.pyplot as plt


class SDFNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d_upsampling_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Upsample(scale_factor=2, mode='trilinear')
        )

        self.conv3d_transpose_conv_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=5, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.Conv3d(in_channels=5, out_channels=10, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.Conv3d(in_channels=10, out_channels=20, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.ConvTranspose3d(in_channels=20, out_channels=10, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(in_channels=10, out_channels=5, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(in_channels=5, out_channels=1, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

    def forward(self, x):
        logits = self.conv3d_transpose_conv_1(x)
        return logits


if __name__ == "__main__":
    start_time = time.perf_counter()

    print("========================================SDF Learner========================================================")
    parser = argparse.ArgumentParser(prog='SDF Learner',
                                     description='Learns the curvature of a signed distance field.')
    parser.add_argument('-o', '--compute_one', help='Compute points of high curvature for one given bin file.')
    parser.add_argument('-a', '--compute_all', action='store', const='Set', nargs='?',
                        help='Compute points of high curvature for all SDF files inside folder.')
    parser.add_argument('-v', '--visualize', help='Visualize points of high curvature for one given bin or csv file.')
    parser.add_argument('-t', '--train', action='store', const='Set', nargs='?',
                        help='Train the neural network using provided samples and labels.')
    args = parser.parse_args()

    point_size = 5
    tolerance = 2000
    percentage = 0.05
    epsilon = 0.1
    start_sample_num = 0
    sample_num = 1000
    in_folder = 'in/'
    in_file_prefix = 'sample'
    in_file_postfix = '_subdiv'
    in_file_extension = '.bin'
    out_folder = 'out/'
    out_file_prefix = 'sample'
    out_file_postfix = ''
    out_file_extension = '.bin'

    print('=> Parameters:')
    print('   Compute one:         ' + str(args.compute_one))
    print('   Compute all:         ' + str(args.compute_all == 'Set'))
    print('   Visualize:           ' + str(args.visualize))
    print('   Train model:         ' + str(args.train == 'Set'))
    time.sleep(2)
    print('')

    if args.compute_one is not None:
        i_path = f'{in_folder}{in_file_prefix}{str(args.compute_one).zfill(6)}{in_file_postfix}{in_file_extension}'
        o_path = f'{out_folder}{out_file_prefix}{str(args.compute_one).zfill(6)}{out_file_postfix}{out_file_extension}'
        sdf_reader = SDFReader(i_path)
        points = sdf_reader.read_points_from_bin(False)
        sdf_curvature = SDFCurvature(epsilon, tolerance, percentage)
        curvatures, sorted_samples = sdf_curvature.calculate_curvature(points)
        points_of_interest = sdf_curvature.classify_points(curvatures, sorted_samples)
        sdf_writer = SDFWriter(o_path)
        sdf_writer.write_points(points_of_interest)

    if args.compute_all is not None:
        for i in range(start_sample_num, sample_num):
            print(f'=> Computing sample {i + 1}')
            i_path = f'{in_folder}{in_file_prefix}{str(i).zfill(6)}{in_file_postfix}{in_file_extension}'
            o_path = f'{out_folder}{out_file_prefix}{str(i).zfill(6)}{out_file_postfix}{out_file_extension}'
            sdf_reader = SDFReader(i_path)
            points = sdf_reader.read_points_from_bin(False, False)
            sdf_curvature = SDFCurvature(epsilon, tolerance, percentage)
            curvatures, sorted_samples = sdf_curvature.calculate_curvature(points, False)
            points_of_interest = sdf_curvature.classify_points(curvatures, sorted_samples, False)
            sdf_writer = SDFWriter(o_path)
            sdf_writer.write_points(points_of_interest, False)

    if args.visualize is not None:
        i_path = str(args.visualize)
        sdf_reader = SDFReader(i_path)
        sdf_visualizer = SDFVisualizer(point_size)
        folder = i_path.split('/')[0]
        if folder == 'in':
            points = sdf_reader.read_points_from_bin(False)
            sdf_curvature = SDFCurvature(epsilon, tolerance, percentage)
            curvatures, sorted_samples = sdf_curvature.calculate_curvature(points)
            points_of_interest = sdf_curvature.classify_points(curvatures, sorted_samples)
            sdf_visualizer.plot_points(points_of_interest, curvatures)
        elif folder == 'out' or folder == 'pred':
            points_of_interest = sdf_reader.read_points_from_bin(True)
            sdf_visualizer.plot_points(points_of_interest, np.zeros(0))
        else:
            print(f'Invalid folder \'{folder}\' found.')

    if args.train:
        full_dataset = SDFDataset('in\\', 'out\\', sample_num)
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])

        # Hyper-parameters of training.
        epochs = 20
        learning_rate = 0.001
        batch_size = 32

        # Initialize train + validation + test data loader with given batch size.
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # Get dimension of input data (most likely always 64 * 64 * 64).
        train_features, _ = next(iter(train_dataloader))
        dim_x, dim_y, dim_z = train_features.size()[2], train_features.size()[3], train_features.size()[4]

        # Check if we have GPU available to run tensors on.
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        model = SDFNeuralNetwork().to(device)

        print(f'=> Starting training...')
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        loss_bce = nn.BCEWithLogitsLoss().to(device)
        train_losses = []
        test_losses = []
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        for t in range(epochs):
            print(f'=> Epoch ({t + 1})')

            # Training loop
            model.train()
            train_loss = 0
            for batch, (X, y) in enumerate(train_dataloader):
                # Compute prediction of current model and compute loss
                prediction = model(X)
                train_loss = loss_bce(prediction, y)

                # Do backpropagation
                train_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                print(f'   BCE loss batch ({batch + 1}): {train_loss.item()}')
            train_losses.append(train_loss.item())

            # Test loop
            model.eval()
            test_loss = 0
            accuracy = 0.0
            precision = 0.0
            recall = 0.0
            f1_score = 0.0
            accuracy_metric = BinaryAccuracy().to(device)
            precision_metric = BinaryPrecision().to(device)
            recall_metric = BinaryRecall().to(device)
            f1_metric = BinaryF1Score().to(device)
            sigmoid = nn.Sigmoid().to(device)
            with torch.no_grad():
                for X, y in test_dataset:
                    # Predict output of one test sample
                    prediction = model(X.reshape((1, 1, dim_x, dim_y, dim_z))).squeeze()
                    test_loss += loss_bce(prediction, y.squeeze()).item()

                    # Update metrics (accuracy, precision, recall, f1) of test samples
                    prediction = sigmoid(prediction)
                    accuracy_metric.update(prediction.reshape((dim_x ** 3)), y.reshape((dim_x ** 3)))
                    precision_metric.update(prediction.reshape((dim_x ** 3)), y.reshape((dim_x ** 3)))
                    recall_metric.update(prediction.reshape((dim_x ** 3)), y.reshape((dim_x ** 3)).int())
                    f1_metric.update(prediction.reshape((dim_x ** 3)), y.reshape((dim_x ** 3)))
            accuracy = accuracy_metric.compute().item()
            precision = precision_metric.compute().item()
            recall = recall_metric.compute().item()
            f1_score = f1_metric.compute().item()
            test_loss /= len(test_dataset)
            print(f'   => Test set summary:\n'
                  f'    - Accuracy:  {accuracy * 100:.2f}% ({accuracy})\n'
                  f'    - Precision: {precision * 100:.2f}% ({precision})\n'
                  f'    - Recall:    {recall * 100:.2f}% ({recall})\n'
                  f'    - F1 score:  {f1_score * 100:.2f}% ({f1_score})\n'
                  f'    - BCE loss:  {test_loss}\n')
            print('')
            test_losses.append(test_loss)
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1_score)

        # TODO: save some predicted samples to pred/ folder (to visualize later)
        sigmoid = nn.Sigmoid()
        prediction = sigmoid(model(full_dataset[0][0].reshape((1, 1, 64, 64, 64))))
        sdf_visualizer = SDFVisualizer(point_size)
        sdf_visualizer.plot_tensor(prediction.squeeze(), dim_x)

        # plt.plot(accuracy_list, color='blue', label='Accuracy')
        plt.plot(precision_list, color='green', label='Precision')
        plt.plot(recall_list, color='blue', label='Recall')
        plt.plot(f1_list, color='red', label='F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('Metric')
        plt.title('Metrics over epochs')
        # plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
        plt.legend()
        plt.show()

        plt.plot(train_losses, color='blue')
        plt.plot(test_losses, color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Loss / accuracy')
        plt.title('BCE loss / test accuracy over epochs')
        plt.show()

    end_time = time.perf_counter()
    print(f'\nFinished in {int((end_time - start_time) / 60)} minutes, {(end_time - start_time) % 60:.4f} seconds.')
