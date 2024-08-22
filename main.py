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
import itertools
import trimesh
import mesh_to_sdf


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
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.Upsample(scale_factor=8, mode='trilinear')
        )

        self.conv3d_transpose_conv_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.Conv3d(in_channels=2, out_channels=4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.Conv3d(in_channels=4, out_channels=8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(in_channels=8, out_channels=4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(in_channels=4, out_channels=2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(in_channels=2, out_channels=1, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.conv3d_transpose_conv_2 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.Conv3d(in_channels=4, out_channels=8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.ConvTranspose3d(in_channels=8, out_channels=4, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=4, out_channels=1, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1))
        )

        self.conv3d_deconvolution_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.Conv3d(in_channels=2, out_channels=4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.Conv3d(in_channels=4, out_channels=8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), return_indices=True),
            nn.MaxUnpool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        )

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels=4, out_channels=4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(in_channels=4, out_channels=8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv5 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv6 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv7 = nn.Conv3d(in_channels=16, out_channels=8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv8 = nn.Conv3d(in_channels=8, out_channels=4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv9 = nn.Conv3d(in_channels=4, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.deconv1 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1))
        self.deconv2 = nn.ConvTranspose3d(in_channels=8, out_channels=4, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()
        self.leakyReLU = nn.LeakyReLU()
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), return_indices=True)

    def u_net(self, x):
        logits = self.conv1(x)
        logits = self.ReLU(logits)
        logits = self.conv2(logits)
        logits = self.ReLU(logits)
        skip1 = torch.clone(logits)
        logits, _ = self.max_pool(logits)
        logits = self.conv3(logits)
        logits = self.ReLU(logits)
        logits = self.conv4(logits)
        logits = self.ReLU(logits)
        skip2 = torch.clone(logits)
        logits, _ = self.max_pool(logits)
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

    def forward(self, x):
        logits = self.conv3d_upsampling_1(x)

        return logits


if __name__ == "__main__":
    start_time = time.perf_counter()

    print("========================================SDF Learner========================================================")
    parser = argparse.ArgumentParser(prog='SDF Learner',
                                     description='Learns the curvature of a signed distance field.')
    parser.add_argument('-go', '--generate_one', help='Generate SDF samples from meshes.')
    parser.add_argument('-ga', '--generate_all', action='store', const='Set', nargs='?',
                        help='Generate SDF samples for all meshes inside folder.')
    parser.add_argument('-co', '--compute_one', help='Compute points of high curvature for one given bin file.')
    parser.add_argument('-ca', '--compute_all', action='store', const='Set', nargs='?',
                        help='Compute points of high curvature for all SDF files inside folder.')
    parser.add_argument('-v', '--visualize', help='Visualize points of high curvature for one given bin or csv file.')
    parser.add_argument('-t', '--train', action='store', const='Set', nargs='?',
                        help='Train the neural network using provided samples and labels.')
    args = parser.parse_args()

    point_size = 5
    tolerance = 0.5
    lower_percentile = 99.5
    upper_percentile = 100.0
    epsilon = 0.1
    start_sample_num = 0
    sample_num = 1000
    sample_dimension = 64
    prediction_num = 3
    in_folder = 'in/'
    in_file_prefix = 'sample'
    in_file_postfix = '_subdiv'
    in_file_extension = '.ply'
    out_folder = 'out/'
    out_file_prefix = 'sample'
    out_file_postfix = ''
    out_file_extension = '.bin'
    sample_folder = 'samples/'
    sample_file_prefix = 'sample'
    sample_file_postfix = '_subdiv'
    sample_file_extension = '.bin'
    pred_folder = 'pred/'
    pred_file_extension = '.bin'

    print('=> Parameters:')
    print('   Generate one:        ' + str(args.generate_one))
    print('   Generate all:        ' + str(args.generate_all == 'Set'))
    print('   Compute one:         ' + str(args.compute_one))
    print('   Compute all:         ' + str(args.compute_all == 'Set'))
    print('   Visualize:           ' + str(args.visualize))
    print('   Train model:         ' + str(args.train == 'Set'))
    time.sleep(2)
    print('')

    if args.generate_one is not None:
        i_path = f'{in_folder}{in_file_prefix}{str(args.generate_one).zfill(6)}{in_file_postfix}{in_file_extension}'
        o_path = f'{sample_folder}{sample_file_prefix}{str(args.generate_one).zfill(6)}{sample_file_postfix}{sample_file_extension}'
        mesh = trimesh.load(i_path)
        x = np.linspace(0, sample_dimension - 1, sample_dimension)
        y = np.linspace(0, sample_dimension - 1, sample_dimension)
        z = np.linspace(0, sample_dimension - 1, sample_dimension)
        points = np.array(list(itertools.product(x, y, z)))
        print('=> Generating sample out of mesh...')
        sdf = mesh_to_sdf.mesh_to_sdf(mesh, points, surface_point_method='scan', sign_method='depth')
        file = open(o_path, 'wb')
        sdf.tofile(file)
        file.close()

    if args.generate_all is not None:
        x = np.linspace(0, sample_dimension - 1, sample_dimension)
        y = np.linspace(0, sample_dimension - 1, sample_dimension)
        z = np.linspace(0, sample_dimension - 1, sample_dimension)
        points = np.array(list(itertools.product(x, y, z)))
        for i in range(start_sample_num, sample_num):
            print(f'=> Generating sample {i + 1}')
            i_path = f'{in_folder}{in_file_prefix}{str(i).zfill(6)}{in_file_postfix}{in_file_extension}'
            o_path = f'{sample_folder}{sample_file_prefix}{str(i).zfill(6)}{sample_file_postfix}{sample_file_extension}'
            mesh = trimesh.load(i_path)
            sdf = mesh_to_sdf.mesh_to_sdf(mesh, points, surface_point_method='scan', sign_method='depth')
            file = open(o_path, 'wb')
            sdf.tofile(file)
            file.close()

    if args.compute_one is not None:
        i_path = f'{sample_folder}{sample_file_prefix}{str(args.compute_one).zfill(6)}{sample_file_postfix}{sample_file_extension}'
        o_path = f'{out_folder}{out_file_prefix}{str(args.compute_one).zfill(6)}{out_file_postfix}{out_file_extension}'
        sdf_reader = SDFReader(i_path)
        points = sdf_reader.read_points_from_bin(False)
        sdf_curvature = SDFCurvature(epsilon, tolerance, lower_percentile, upper_percentile)
        curvatures, sorted_samples = sdf_curvature.calculate_curvature(points)
        points_of_interest = sdf_curvature.classify_points(curvatures, sorted_samples)
        sdf_writer = SDFWriter(o_path)
        sdf_writer.write_points(points_of_interest)

    if args.compute_all is not None:
        for i in range(start_sample_num, sample_num):
            print(f'=> Computing sample {i + 1}')
            i_path = f'{sample_folder}{sample_file_prefix}{str(i).zfill(6)}{sample_file_postfix}{sample_file_extension}'
            o_path = f'{out_folder}{out_file_prefix}{str(i).zfill(6)}{out_file_postfix}{out_file_extension}'
            sdf_reader = SDFReader(i_path)
            points = sdf_reader.read_points_from_bin(False, False)
            sdf_curvature = SDFCurvature(epsilon, tolerance, lower_percentile, upper_percentile)
            curvatures, sorted_samples = sdf_curvature.calculate_curvature(points, False)
            points_of_interest = sdf_curvature.classify_points(curvatures, sorted_samples, False)
            sdf_writer = SDFWriter(o_path)
            sdf_writer.write_points(points_of_interest, False)

    if args.visualize is not None:
        i_path = str(args.visualize)
        sdf_reader = SDFReader(i_path)
        sdf_visualizer = SDFVisualizer(point_size)
        folder = i_path.split('/')[0]
        if folder == 'in' or folder == 'samples':
            points = sdf_reader.read_points_from_bin(False)
            sdf_curvature = SDFCurvature(epsilon, tolerance, lower_percentile, upper_percentile)
            curvatures, sorted_samples = sdf_curvature.calculate_curvature(points)
            points_of_interest = sdf_curvature.classify_points(curvatures, sorted_samples)
            sdf_visualizer.plot_points(points, points_of_interest, curvatures)
        elif folder == 'out' or folder == 'pred':
            points_of_interest = sdf_reader.read_points_from_bin(True)
            sdf_visualizer.plot_points(points_of_interest, points_of_interest, np.zeros(0))
        else:
            print(f'Invalid folder \'{folder}\' found.')

    if args.train:
        full_dataset = SDFDataset('samples\\', 'out\\', sample_num)
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])

        # Hyper-parameters of training.
        epochs = 20
        learning_rate = 0.0005
        batch_size = 16

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
        weights = torch.tensor([200])
        loss_bce = nn.BCEWithLogitsLoss(pos_weight=weights).to(device)
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
                    accuracy_metric.update(prediction.reshape((dim_x ** 3)), y.reshape((dim_x ** 3)).int())
                    precision_metric.update(prediction.reshape((dim_x ** 3)), y.reshape((dim_x ** 3)).int())
                    recall_metric.update(prediction.reshape((dim_x ** 3)), y.reshape((dim_x ** 3)).int())
                    f1_metric.update(prediction.reshape((dim_x ** 3)), y.reshape((dim_x ** 3)).int())
            accuracy = accuracy_metric.compute().item()
            precision = precision_metric.compute().item()
            recall = recall_metric.compute().item()
            f1_score = f1_metric.compute().item()
            test_loss /= len(test_dataset)
            # TODO: log this output to file in pred/ folder
            print(f'   => Test set summary:\n'
                  f'    - Accuracy:  {accuracy * 100:.2f}% ({accuracy})\n'
                  f'    - Precision: {precision * 100:.2f}% ({precision})\n'
                  f'    - Recall:    {recall * 100:.2f}% ({recall})\n'
                  f'    - F1 score:  {f1_score * 100:.2f}% ({f1_score})\n'
                  f'    - BCE loss:  {test_loss}\n')
            test_losses.append(test_loss)
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1_score)
            accuracy_metric.reset()
            precision_metric.reset()
            recall_metric.reset()
            f1_metric.reset()

        # Save some predicted samples to pred/ folder (to visualize later)
        sigmoid = nn.Sigmoid()
        date = time.strftime('%Y%m%d%H%M')
        for i in range(prediction_num):
            o_path = f'{pred_folder}{date}_{str(i).zfill(6)}{pred_file_extension}'
            prediction = sigmoid(model(full_dataset[i][0].reshape((1, 1, 64, 64, 64))))
            sdf_writer = SDFWriter(o_path)
            prediction_conv = prediction.squeeze().round().numpy(force=True).astype(np.int32)
            sdf_writer.write_points(prediction_conv)

        # plt.plot(accuracy_list, color='yellow', label='Accuracy')
        plt.plot(precision_list, color='green', label='Precision')
        plt.plot(recall_list, color='blue', label='Recall')
        plt.plot(f1_list, color='red', label='F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('Metric')
        plt.title('Metrics over epochs')
        plt.legend()
        plt.savefig(f'{pred_folder}{date}_metrics.png')
        plt.show()

        plt.plot(train_losses, color='blue')
        plt.plot(test_losses, color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('BCE loss over epochs')
        plt.savefig(f'{pred_folder}{date}_loss.png')
        plt.show()

    end_time = time.perf_counter()
    print(f'\nFinished in {int((end_time - start_time) / 60)} minutes, {(end_time - start_time) % 60:.4f} seconds.')
