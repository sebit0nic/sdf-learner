"""
File name: main.py
Author: Sebastian Lackner
Version: 1.0
Description: Main entry point of program, handling of user parameters
"""

import numpy as np
import argparse
import time
import itertools
import trimesh
import mesh_to_sdf

from SDFFileHandler import SDFReader
from SDFFileHandler import SDFWriter
from SDFUtil import SDFCurvature
from SDFVisualizer import SDFVisualizer
from SDFTraining import SDFTrainer
from sdf import Sdf3D


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
    parser.add_argument('-vo', '--visualize_obj', help='Visualize obj file.')
    parser.add_argument('-t', '--train', help='Train the neural network using some predefined model.')
    parser.add_argument('-s', '--grid_search', action='store', const='Set', nargs='?',
                        help='Grid search over hyperparameters before final training.')

    args = parser.parse_args()

    point_size = 10
    epsilon = 0.001
    percentage = 20
    resolution = 0.5
    threshold = 0.5
    start_sample_num = 0
    sample_num = 1000
    sample_dim = 64
    in_folder = 'in/'
    in_file_prefix = 'sample'
    in_file_postfix = '_subdiv'
    in_file_extension = '.ply'
    out_folder = 'out/'
    out_file_prefix = 'sample'
    out_file_postfix = ''
    out_file_extension = '.bin'
    sample_folder = 'in/'
    sample_file_prefix = 'sample'
    sample_file_postfix = '_subdiv'
    sample_file_extension = '.bin'

    print('=> Parameters:')
    print('   Generate one: ' + str(args.generate_one))
    print('   Generate all: ' + str(args.generate_all == 'Set'))
    print('   Compute one:  ' + str(args.compute_one))
    print('   Compute all:  ' + str(args.compute_all == 'Set'))
    print('   Visualize:    ' + str(args.visualize))
    print('   Train:        ' + str(args.train))
    print('   Grid search:  ' + str(args.grid_search == 'Set'))
    time.sleep(2)
    print('')

    # Do conversion of one triangle mesh to SDF (experimental)
    if args.generate_one is not None:
        i_path = f'{in_folder}{in_file_prefix}{str(args.generate_one).zfill(6)}{in_file_postfix}{in_file_extension}'
        o_path = (f'{sample_folder}{sample_file_prefix}{str(args.generate_one).zfill(6)}{sample_file_postfix}'
                  f'{sample_file_extension}')
        mesh = trimesh.load(i_path)
        x = np.linspace(0, sample_dim - 1, sample_dim)
        y = np.linspace(0, sample_dim - 1, sample_dim)
        z = np.linspace(0, sample_dim - 1, sample_dim)
        points = np.array(list(itertools.product(x, y, z)))
        print('=> Generating sample out of mesh...')
        sdf = mesh_to_sdf.mesh_to_sdf(mesh, points, surface_point_method='scan', sign_method='depth')
        file = open(o_path, 'wb')
        sdf.tofile(file)
        file.close()

    # Do conversion of multiple triangle meshes to SDFs (experimental)
    if args.generate_all is not None:
        x = np.linspace(0, sample_dim - 1, sample_dim)
        y = np.linspace(0, sample_dim - 1, sample_dim)
        z = np.linspace(0, sample_dim - 1, sample_dim)
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

    # Compute the ground truth out of one SDF
    if args.compute_one is not None:
        i_path = (f'{in_folder}{sample_file_prefix}{str(args.compute_one).zfill(6)}{sample_file_postfix}'
                  f'{sample_file_extension}')
        o_path = f'{out_folder}{out_file_prefix}{str(args.compute_one).zfill(6)}{out_file_postfix}{out_file_extension}'
        sdf_reader = SDFReader(i_path)
        points = sdf_reader.read_points_from_bin(False)
        sdf = Sdf3D(points, np.array((sample_dim / 2, sample_dim / 2, sample_dim / 2)), resolution)
        sdf_curvature = SDFCurvature(epsilon, percentage, sdf.dimensions[0])
        _, sorted_samples = sdf_curvature.calculate_curvature(sdf)
        points_of_interest = sdf_curvature.classify_points(sorted_samples)
        sdf_writer = SDFWriter(o_path)
        sdf_writer.write_points(points_of_interest)

    # Compute the ground truths out of multiple SDFs
    if args.compute_all is not None:
        for i in range(start_sample_num, sample_num):
            print(f'=> Computing sample {i + 1}')
            i_path = f'{in_folder}{sample_file_prefix}{str(i).zfill(6)}{sample_file_postfix}{sample_file_extension}'
            o_path = f'{out_folder}{out_file_prefix}{str(i).zfill(6)}{out_file_postfix}{out_file_extension}'
            sdf_reader = SDFReader(i_path)
            points = sdf_reader.read_points_from_bin(False, False)
            sdf = Sdf3D(points, np.array((sample_dim / 2, sample_dim / 2, sample_dim / 2)), resolution)
            sdf_curvature = SDFCurvature(epsilon, percentage, sdf.dimensions[0])
            _, sorted_samples = sdf_curvature.calculate_curvature(sdf, False)
            points_of_interest = sdf_curvature.classify_points(sorted_samples, False)
            sdf_writer = SDFWriter(o_path)
            sdf_writer.write_points(points_of_interest, False)

    # Visualize SDF or ground truth as 3D point cloud
    if args.visualize is not None:
        i_path = str(args.visualize)
        if args.visualize_obj is not None:
            i_obj_path = f'{in_folder}{sample_file_prefix}{str(args.visualize_obj).zfill(6)}{sample_file_postfix}.obj'
        else:
            i_obj_path = ''
        sdf_reader = SDFReader(i_path)
        sdf_visualizer = SDFVisualizer(point_size)
        folder = i_path.split('/')[0]
        if folder == 'in' or folder == 'in_v1' or folder == 'in_v2' or folder == 'samples':
            points = sdf_reader.read_points_from_bin(False)
            sdf = Sdf3D(points, np.array((sample_dim / 2, sample_dim / 2, sample_dim / 2)), resolution)
            sdf_curvature = SDFCurvature(epsilon, percentage, sdf.dimensions[0])
            curvatures, sorted_samples = sdf_curvature.calculate_curvature(sdf)
            points_of_interest = sdf_curvature.classify_points(sorted_samples)
            sdf_visualizer.plot_points(points_of_interest, curvatures, i_obj_path)
        elif folder == 'out' or folder == 'out_v1' or folder == 'out_v2' or folder == 'pred':
            points_of_interest = sdf_reader.read_points_from_bin(True)
            sdf_visualizer.plot_points(points_of_interest, np.zeros(0), i_obj_path)
        else:
            print(f'Invalid folder \'{folder}\' found.')

    # Train model on SDF data and ground truth
    if args.train is not None:
        trainer = SDFTrainer(str(args.train), args.grid_search, in_folder, out_folder)
        trainer.train()

    end_time = time.perf_counter()
    print(f'\nFinished in {int((end_time - start_time) / 60)} minutes, {(end_time - start_time) % 60:.4f} seconds.')
