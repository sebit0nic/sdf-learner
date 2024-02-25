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
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128 ** 3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128 ** 3),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == "__main__":
    print("========================================SDF Learner========================================================")
    parser = argparse.ArgumentParser(prog='SDF Learner',
                                     description='Learns the curvature of a signed distance field.')
    parser.add_argument('-i', '--sdf_in', help='Binary file that contains a signed distance field.')
    parser.add_argument('-o', '--sdf_out', help='File where estimated points of high curvature are written to.')
    parser.add_argument('-c', '--sdf_config', help='File where configuration of signed distance field is stored.')
    parser.add_argument('-d', '--curvature', action='store', const='Set', nargs='?',
                        help='Calculate the derivative and curvature of the given SDF samples.')
    parser.add_argument('-p', '--pyvista', action='store', const='Set', nargs='?',
                        help='Show graphics as pyvista graph.')
    parser.add_argument('-t', '--train', action='store', const='Set', nargs='?',
                        help='Train the neural network using provided samples and labels.')
    args = parser.parse_args()

    point_size = 5
    tolerance = 0.01
    percentage = 0.1
    if args.sdf_config is not None:
        sdf_reader = SDFReader(args.sdf_config)
        point_size, tolerance, percentage = sdf_reader.read_configuration()
    print('=> Parameters:')
    print('   SDF input file:      ' + str(args.sdf_in))
    print('   SDF output file:     ' + str(args.sdf_out))
    print('   Point size:          ' + str(point_size))
    print('   Tolerance:           ' + str(tolerance))
    print('   Percentage:          ' + str(percentage))
    print('   Calculate curvature: ' + str(args.curvature == 'Set'))
    print('   Show graph:          ' + str(args.pyvista == 'Set'))
    print('   Train model:         ' + str(args.train == 'Set'))
    time.sleep(2)
    print('')

    samples = []
    if args.sdf_in is not None:
        sdf_reader = SDFReader(args.sdf_in)
        samples = sdf_reader.read_samples()

    if args.curvature:
        sdf_curvature = SDFCurvature(0.1, tolerance, percentage)
        samples, sorted_samples = sdf_curvature.calculate_curvature(samples)
        samples = sdf_curvature.classify_samples(samples, sorted_samples)

    if args.sdf_out is not None:
        sdf_writer = SDFWriter(args.sdf_out)
        sdf_writer.write_samples(samples)

    if args.pyvista:
        sdf_visualizer = SDFVisualizer(point_size)
        sdf_visualizer.plot_samples(samples)

    if args.train:
        full_dataset = SDFDataset('in\\', 'out\\')
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [2, 1])

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

        epochs = 5
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        loss_nll = nn.CrossEntropyLoss()
        for t in range(epochs):
            model.train()
            for batch, (X, y) in enumerate(train_dataloader):
                pred = model(X)
                pred = nn.Softmax(dim=1)(pred)
                loss = loss_nll(pred, y)
                print(X)
                print(pred)
                print(y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss = loss.item()
                print(f"loss: {loss:>7f}")
