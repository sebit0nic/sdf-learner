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
import math

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
    parser.add_argument('-rc', '--read_curvature', help='Read points of high curvature for one given bin file.')
    parser.add_argument('-v', '--visualize', help='Visualize points of high curvature for one given bin or csv file.')
    parser.add_argument('-vo', '--visualize_obj', help='Visualize obj file.')
    parser.add_argument('-t', '--train', help='Train the neural network using some predefined model.')
    parser.add_argument('-s', '--grid_search', action='store', const='Set', nargs='?',
                        help='Grid search over hyperparameters before final training.')

    args = parser.parse_args()

    point_size = 10
    epsilon = 0.01
    percentage = 25
    resolution = 0.5
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
    curvature_folder = 'curvature/'
    curvature_file_prefix = 'sample'
    curvature_file_postfix = '_subdiv'
    curvature_file_extension = '.bin'

    print('=> Parameters:')
    print('   Generate one: ' + str(args.generate_one))
    print('   Generate all: ' + str(args.generate_all == 'Set'))
    print('   Compute one:  ' + str(args.compute_one))
    print('   Compute all:  ' + str(args.compute_all == 'Set'))
    print('   Read curve:   ' + str(args.read_curvature))
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
        sdf = Sdf3D(points, np.array((sample_dim / 2, sample_dim / 2, sample_dim / 2)), resolution, epsilon)
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
            sdf = Sdf3D(points, np.array((sample_dim / 2, sample_dim / 2, sample_dim / 2)), resolution, epsilon)
            sdf_curvature = SDFCurvature(epsilon, percentage, sdf.dimensions[0])
            _, sorted_samples = sdf_curvature.calculate_curvature(sdf, False)
            points_of_interest = sdf_curvature.classify_points(sorted_samples, False)
            sdf_writer = SDFWriter(o_path)
            sdf_writer.write_points(points_of_interest, False)

    if args.read_curvature is not None:
        i_c_path = (f'{curvature_folder}{curvature_file_prefix}{str(args.read_curvature).zfill(6)}'
                    f'{curvature_file_postfix}{curvature_file_extension}')
        i_d_path = (f'{in_folder}{sample_file_prefix}{str(args.read_curvature).zfill(6)}{sample_file_postfix}'
                    f'{sample_file_extension}')
        if args.visualize_obj is not None:
            i_obj_path = f'{in_folder}{sample_file_prefix}{str(args.visualize_obj).zfill(6)}{sample_file_postfix}.obj'
        else:
            i_obj_path = ''

        # Read in vertex coordinates and corresponding curvature from file (see mesh.py for generation)
        file = open(i_c_path)
        data = np.fromfile(file)
        curvature_data = np.reshape(data, (data.shape[0] // 4, 4))
        curvature_data = np.hstack((curvature_data, np.zeros((data.shape[0] // 4, 1))))
        file.close()
        file = open(i_d_path)
        distances = np.fromfile(file, dtype=np.float32).reshape((sample_dim, sample_dim, sample_dim))
        file.close()

        # Take absolute and sort curvatures of vertices in descending order to later iterate over them
        # TODO: do this already in mesh.py
        for c in curvature_data:
            c[4] = c[3]
            c[3] = abs(c[3])
        sorted_curvature = curvature_data[curvature_data[:, 3].argsort()[::-1]]

        # Classify a certain percentage of vertices of the mesh as high curvature points in the SDF
        num_curvatures = round(np.shape(sorted_curvature)[0] * (percentage / 100))
        points_of_interest = np.zeros((sample_dim, sample_dim, sample_dim), dtype=np.int32)
        curvatures = np.zeros((sample_dim, sample_dim, sample_dim))
        for i in range(num_curvatures):
            v = sorted_curvature[i]

            # Get coordinates of neighbouring points in SDF based on mesh vertex coordinate
            f0 = math.floor(v[0])
            c0 = math.ceil(v[0])
            f1 = math.floor(v[1])
            c1 = math.ceil(v[1])
            f2 = math.floor(v[2])
            c2 = math.ceil(v[2])
            cur_dist = math.inf
            cur_z = 0
            cur_y = 0
            cur_x = 0
            loc_array = [[f0, f1, f2], [f0, f1, c2], [f0, c1, f2], [f0, c1, c2], [c0, f1, f2], [c0, f1, c2],
                         [c0, c1, f2], [c0, c1, c2]]
            dist_array = [distances[f0, f1, f2], distances[f0, f1, c2], distances[f0, c1, f2], distances[f0, c1, c2],
                          distances[c0, f1, f2], distances[c0, f1, c2], distances[c0, c1, f2], distances[c0, c1, c2]]

            # Find neighbouring SDF point with the smallest distance to surface to place point as close as possible
            for k in range(len(loc_array)):
                if abs(dist_array[k]) < cur_dist:
                    cur_dist = abs(dist_array[k])
                    cur_z = loc_array[k][0]
                    cur_y = loc_array[k][1]
                    cur_x = loc_array[k][2]

            # Classify corresponding point in SDF as high curvature point
            points_of_interest[cur_z, cur_y, cur_x] = 1
            curvatures[cur_z, cur_y, cur_x] = sorted_curvature[i][4]
        sdf_visualizer = SDFVisualizer(point_size)
        sdf_visualizer.plot_points(points_of_interest, curvatures, i_obj_path)

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
            sdf = Sdf3D(points, np.array((sample_dim / 2, sample_dim / 2, sample_dim / 2)), resolution, epsilon)
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
