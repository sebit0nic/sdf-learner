import numpy as np
import torch.utils.data
import torchinfo
from torch.utils.data import DataLoader
from torch import nn
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryConfusionMatrix
from torchmetrics.classification import BinaryJaccardIndex

import SDFTraining
from SDFFileHandler import SDFReader
from SDFFileHandler import SDFWriter
from SDFUtil import SDFCurvature
from SDFVisualizer import SDFVisualizer
from SDFDataset import SDFDataset
from SDFTraining import DiceLoss, TverskyLoss, FocalTverskyLoss
from SDFTraining import SDFUnetLevel2, SDFUnetLevel3, SDFTrainer
import argparse
import time
import matplotlib.pyplot as plt
import seaborn as sn
import itertools
import trimesh
import mesh_to_sdf


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
    parser.add_argument('-tu2', '--train_unet_2', action='store', const='Set', nargs='?',
                        help='Train the neural network using the U-Net model with 2 levels.')
    parser.add_argument('-tu3', '--train_unet_3', action='store', const='Set', nargs='?',
                        help='Train the neural network using the U-Net model with 3 levels.')

    args = parser.parse_args()

    point_size = 5
    tolerance = 0.5
    lower_percentile = 99.5
    upper_percentile = 100.0
    epsilon = 0.1
    start_sample_num = 0
    sample_num = 1000
    sample_dimension = 64
    prediction_num = 5
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
    print('   Train U-Net Lvl 2:   ' + str(args.train_unet_2 == 'Set'))
    print('   Train U-Net Lvl 3:   ' + str(args.train_unet_3 == 'Set'))
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

    if args.train_unet_2:
        trainer = SDFTrainer('u2', True)
        trainer.train()

    if args.train:
        # TODO: implement grid search (per network structure)
        full_dataset = SDFDataset('samples\\', 'out\\', sample_num)
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])

        # Hyper-parameters of training.
        epochs = 30
        learning_rate = 0.001
        batch_size = 8

        # Initialize train + validation + test data loader with given batch size.
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Get dimension of input data (most likely always 64 * 64 * 64).
        train_features, _ = next(iter(train_dataloader))
        dim_x, dim_y, dim_z = train_features.size()[2], train_features.size()[3], train_features.size()[4]

        # Check if we have GPU available to run tensors on.
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        loss_functions = [DiceLoss(),
                          TverskyLoss(0.5),
                          TverskyLoss(0.1),
                          TverskyLoss(0.9),
                          FocalTverskyLoss(0.5, 2),
                          FocalTverskyLoss(0.1, 2),
                          FocalTverskyLoss(0.9, 2)]

        iteration = 1
        for func in loss_functions:
            # weights = torch.tensor([1])
            # loss_function = nn.BCEWithLogitsLoss(pos_weight=weights).to(device)
            loss_function = func.to(device)
            model = SDFUnetLevel3().to(device)
            model_description = torchinfo.summary(model, (1, 1, 64, 64, 64), verbose=0)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            print(f'=> Starting training {iteration}...')
            date = time.strftime('%Y%m%d%H%M')
            train_losses = []
            test_losses = []
            accuracy_list = []
            precision_list = []
            recall_list = []
            f1_list = []
            mIOU_list = []
            confusion_matrix = []
            with open(f'{pred_folder}{date}_log.txt', 'w', encoding='utf-8') as log_file:
                log_str = f'   Epochs:        {epochs}\n' \
                          f'   Learning rate: {learning_rate}\n' \
                          f'   Batch size:    {batch_size}\n' \
                          f'   Optimizer:     {str(optimizer)}\n' \
                          f'   Loss function: {loss_function.__class__.__name__}{vars(loss_function)}\n' \
                          f'{str(model_description)}\n'
                log_file.write(log_str)
                for t in range(epochs):
                    log_str = f'=> Epoch ({t + 1})'
                    log_file.write(log_str)
                    print(log_str)

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
                    accuracy = 0.0
                    precision = 0.0
                    recall = 0.0
                    f1_score = 0.0
                    mIOU = 0.0
                    accuracy_metric = BinaryAccuracy().to(device)
                    precision_metric = BinaryPrecision().to(device)
                    recall_metric = BinaryRecall().to(device)
                    f1_metric = BinaryF1Score().to(device)
                    mIOU_metric = BinaryJaccardIndex().to(device)
                    confusion_metric = BinaryConfusionMatrix(normalize='pred').to(device)
                    sigmoid = nn.Sigmoid().to(device)
                    with torch.no_grad():
                        for X, y in test_dataset:
                            # Predict output of one test sample
                            prediction = model(X.reshape((1, 1, dim_x, dim_y, dim_z)))
                            test_loss += loss_function(prediction, y.unsqueeze(dim=0)).item()

                            # Update metrics (accuracy, precision, recall, f1) of test samples
                            prediction = sigmoid(prediction)
                            prediction_conv = prediction.reshape((dim_x ** 3))
                            label_conv = y.reshape((dim_x ** 3)).int()
                            accuracy_metric.update(prediction_conv, label_conv)
                            precision_metric.update(prediction_conv, label_conv)
                            recall_metric.update(prediction_conv, label_conv)
                            f1_metric.update(prediction_conv, label_conv)
                            mIOU_metric.update(prediction_conv, label_conv)
                            confusion_metric.update(prediction_conv, label_conv)
                    accuracy = accuracy_metric.compute().item()
                    precision = precision_metric.compute().item()
                    recall = recall_metric.compute().item()
                    f1_score = f1_metric.compute().item()
                    mIOU = mIOU_metric.compute().item()
                    test_loss /= len(test_dataset)
                    confusion_matrix = confusion_metric.compute()

                    # Output metrics + write to log file for later
                    log_str = f'   => Test set summary:\n' \
                              f'    - Accuracy:  {accuracy * 100:.2f}% ({accuracy})\n' \
                              f'    - Precision: {precision * 100:.2f}% ({precision})\n' \
                              f'    - Recall:    {recall * 100:.2f}% ({recall})\n' \
                              f'    - F1 score:  {f1_score * 100:.2f}% ({f1_score})\n' \
                              f'    - mIOU:      {mIOU * 100:.2f}% ({mIOU})\n' \
                              f'    - Loss:      {test_loss}\n'
                    print(log_str)
                    log_file.write(log_str)

                    test_losses.append(test_loss)
                    accuracy_list.append(accuracy)
                    precision_list.append(precision)
                    recall_list.append(recall)
                    f1_list.append(f1_score)
                    mIOU_list.append(mIOU)
                    accuracy_metric.reset()
                    precision_metric.reset()
                    recall_metric.reset()
                    f1_metric.reset()
                    mIOU_metric.reset()
                    confusion_metric.reset()

            iteration += 1

            # Save some predicted samples to pred/ folder (to visualize later)
            sigmoid = nn.Sigmoid()
            for i in range(prediction_num):
                o_path = f'{pred_folder}{date}_{str(i).zfill(6)}{pred_file_extension}'
                prediction = sigmoid(model(full_dataset[i][0].reshape((1, 1, 64, 64, 64))))
                sdf_writer = SDFWriter(o_path)
                prediction_conv = prediction.squeeze().round().numpy(force=True).astype(np.int32)
                sdf_writer.write_points(prediction_conv)

            plt.plot(accuracy_list, color='cyan', label='Accuracy')
            plt.plot(mIOU_list, color='orange', label='mIOU')
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

            sn.heatmap(confusion_matrix.numpy(force=True), square=True, xticklabels='01', yticklabels='01', annot=True)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.savefig(f'{pred_folder}{date}_confusion.png')
            plt.show()

    end_time = time.perf_counter()
    print(f'\nFinished in {int((end_time - start_time) / 60)} minutes, {(end_time - start_time) % 60:.4f} seconds.')
