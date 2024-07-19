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


class SDFNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: try with deconvolutional layers?
        self.conv3d_stack = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.conv3d_stack(x)
        return logits


if __name__ == "__main__":
    print("========================================SDF Learner========================================================")
    parser = argparse.ArgumentParser(prog='SDF Learner',
                                     description='Learns the curvature of a signed distance field.')
    parser.add_argument('-o', '--compute_one', help='Compute points of high curvature for one given SDF file.')
    parser.add_argument('-a', '--compute_all', action='store', const='Set', nargs='?',
                        help='Compute points of high curvature for all SDF files inside folder.')
    parser.add_argument('-v', '--visualize', help='Visualize points of high curvature for one given SDF file.')
    parser.add_argument('-t', '--train', action='store', const='Set', nargs='?',
                        help='Train the neural network using provided samples and labels.')
    args = parser.parse_args()

    point_size = 5
    tolerance = 2000
    percentage = 0.05
    epsilon = 0.1
    sample_num = 5
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
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [3, 2])

        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
        train_features, train_labels = next(iter(train_dataloader))
        test_features, test_labels = next(iter(test_dataloader))
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        print(f"Feature batch shape: {test_features.size()}")
        print(f"Labels batch shape: {test_labels.size()}")

        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        model = SDFNeuralNetwork().to(device)
        for name, param in model.named_parameters():
            print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

        epochs = 5
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        loss_ce = nn.CrossEntropyLoss()
        for t in range(epochs):
            print(f'### Epoch ({t + 1}) ###')
            model.train()
            for batch, (X, y) in enumerate(train_dataloader):
                # TODO: flatten prediction from 3D to 1D tensor
                prediction = model(X)
                prediction_normalized = prediction.round()
                y = torch.squeeze(y).long()
                prediction_transformed = torch.zeros((prediction.size()[0], 2, prediction.size()[2], prediction.size()[3], prediction.size()[4])).to(device)
                prediction_transformed[:, 0, :, :, :] = torch.clone(torch.squeeze(prediction))
                prediction_transformed[:, 1, :, :, :] = torch.clone(torch.squeeze(prediction))
                print(f'Prediction size: {prediction.size()}')
                print(f'Prediction normalized size: {prediction_normalized.size()}')
                print(f'Target size: {y.size()}')
                print(f'Number of positive groups in y: {torch.count_nonzero(y)}')
                print(f'Number of positive groups in prediction: {torch.count_nonzero(prediction_normalized)}')
                loss = loss_ce(prediction_transformed, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss = loss.item()
                print(f"loss: {loss:>7f}")
