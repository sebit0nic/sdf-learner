import torch.utils.data
from torch.utils.data import DataLoader
from torch import nn

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
        # TODO: try with deconvolutional layers?
        self.conv3d_single_channel = nn.Sequential(
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

    def forward(self, x):
        logits = self.conv3d_single_channel(x)
        return logits


if __name__ == "__main__":
    print("========================================SDF Learner========================================================")
    parser = argparse.ArgumentParser(prog='SDF Learner',
                                     description='Learns the curvature of a signed distance field.')
    parser.add_argument('-o', '--compute_one', help='Compute points of high curvature for one given SDF file.')
    parser.add_argument('-a', '--compute_all', action='store', const='Set', nargs='?',
                        help='Compute points of high curvature for all SDF files inside folder.')
    parser.add_argument('-vb', '--visualize', help='Visualize points of high curvature for one given SDF file.')
    # TODO: argument to visualize csv file
    parser.add_argument('-t', '--train', action='store', const='Set', nargs='?',
                        help='Train the neural network using provided samples and labels.')
    args = parser.parse_args()

    point_size = 5
    tolerance = 2000
    percentage = 0.05
    epsilon = 0.1
    sample_num = 1000
    in_folder = 'in/'
    in_file_prefix = 'sample'
    in_file_postfix = '_subdiv'
    in_file_extension = '.bin'
    out_folder = 'out/'
    out_file_prefix = 'sample'
    out_file_postfix = ''
    out_file_extension = '.csv'

    print('=> Parameters:')
    print('   Compute one:         ' + str(args.compute_one))
    print('   Compute all:         ' + str(args.compute_all == 'Set'))
    print('   Visualize:           ' + str(args.visualize))
    print('   Train model:         ' + str(args.train == 'Set'))
    time.sleep(2)
    print('')

    samples = []
    if args.compute_one is not None:
        i_path = f'{in_folder}{in_file_prefix}{str(args.compute_one).zfill(6)}{in_file_postfix}{in_file_extension}'
        o_path = f'{out_folder}{out_file_prefix}{str(args.compute_one).zfill(6)}{out_file_postfix}{out_file_extension}'
        sdf_reader = SDFReader(i_path)
        samples, size = sdf_reader.read_points()
        sdf_curvature = SDFCurvature(epsilon, tolerance, percentage)
        samples, sorted_samples = sdf_curvature.calculate_curvature(samples, size)
        samples = sdf_curvature.classify_points(samples, sorted_samples)
        sdf_writer = SDFWriter(o_path, size)
        sdf_writer.write_points(samples)

    if args.compute_all is not None:
        for i in range(sample_num):
            print(f'=> Computing sample {i + 1}')
            i_path = f'{in_folder}{in_file_prefix}{str(i).zfill(6)}{in_file_postfix}{in_file_extension}'
            o_path = f'{out_folder}{out_file_prefix}{str(i).zfill(6)}{out_file_postfix}{out_file_extension}'
            sdf_reader = SDFReader(i_path)
            samples, size = sdf_reader.read_points(False)
            sdf_curvature = SDFCurvature(epsilon, tolerance, percentage)
            samples, sorted_samples = sdf_curvature.calculate_curvature(samples, size, False)
            samples = sdf_curvature.classify_points(samples, sorted_samples, False)
            sdf_writer = SDFWriter(o_path, size)
            sdf_writer.write_points(samples, False)

    if args.visualize is not None:
        i_path = f'{in_folder}{in_file_prefix}{str(args.visualize).zfill(6)}{in_file_postfix}{in_file_extension}'
        sdf_reader = SDFReader(i_path)
        samples, size = sdf_reader.read_points()
        sdf_curvature = SDFCurvature(epsilon, tolerance, percentage)
        samples, sorted_samples = sdf_curvature.calculate_curvature(samples, size)
        samples = sdf_curvature.classify_points(samples, sorted_samples)
        sdf_visualizer = SDFVisualizer(point_size)
        sdf_visualizer.plot_points(samples, size)

    if args.train:
        full_dataset = SDFDataset('in\\', 'out\\', sample_num)
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])

        # Hyper-parameters of training.
        epochs = 1
        learning_rate = 0.001
        batch_size = 20

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

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        loss_bce = nn.BCEWithLogitsLoss()
        train_losses = []
        test_losses = []
        accuracies = []
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
            sigmoid = nn.Sigmoid()
            with torch.no_grad():
                for X, y in test_dataloader:
                    prediction = model(X)
                    test_loss += loss_bce(prediction, y).item()
                    prediction = sigmoid(prediction).round()
                    accuracy += torch.eq(prediction, y).sum().item()
            accuracy /= len(test_dataset) * dim_x * dim_y * dim_z
            test_loss /= len(test_dataloader)
            print(f'   => Test set accuracy: {accuracy}, BCE loss: {test_loss}')
            print('')
            test_losses.append(test_loss)
            accuracies.append(accuracy)

        # TODO: save some predicted samples to pred/ folder (to visualize later)
        # sigmoid = nn.Sigmoid()
        # prediction = sigmoid(model(full_dataset[0][0].reshape((1, 1, 64, 64, 64))))
        # sdf_visualizer = SDFVisualizer(point_size)
        # sdf_visualizer.plot_tensor(prediction.squeeze(), dim_x)

        plt.plot(train_losses, color='blue')
        plt.plot(test_losses, color='green')
        plt.plot(accuracies, color='green', linestyle='dashed')
        plt.xlabel('Epochs')
        plt.ylabel('Loss / accuracy')
        plt.title('BCE loss / test accuracy over epochs')
        plt.show()
