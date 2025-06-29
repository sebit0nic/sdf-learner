"""
File name: mesh.py
Author: Sebastian Lackner
Version: 1.0
Description: Computation of mesh curvature data
"""

import pymesh
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='SDF Learner',
                                     description='Learns the curvature of a signed distance field.')
    parser.add_argument('-co', '--compute_one', help='Compute points of high curvature for one given bin file.')
    parser.add_argument('-ca', '--compute_all', action='store', const='Set', nargs='?',
                        help='Compute points of high curvature for all mesh files inside folder.')

    args = parser.parse_args()

    start_sample_num = 0
    sample_num = 1000
    in_folder = 'in/'
    in_file_prefix = 'sample'
    in_file_postfix = '_subdiv'
    in_file_extension = '.ply'
    curvature_folder = 'curvature/'
    curvature_file_prefix = 'sample'
    curvature_file_postfix = '_subdiv'
    curvature_file_extension = '.bin'

    # Compute the ground truth out of one mesh
    if args.compute_one is not None:
        i_path = f'{in_folder}{in_file_prefix}{str(args.compute_one).zfill(6)}{in_file_postfix}{in_file_extension}'
        o_path = (f'{curvature_folder}{curvature_file_prefix}{str(args.compute_one).zfill(6)}'
                  f'{curvature_file_postfix}{curvature_file_extension}')
        mesh = pymesh.quad_to_tri(pymesh.load_mesh(i_path))
        # TODO: add column for absolute value of curvature
        mesh.add_attribute("vertex_gaussian_curvature")
        curvature = mesh.get_attribute("vertex_gaussian_curvature")
        curvature_data = np.hstack((mesh.vertices, np.reshape(curvature, (mesh.num_nodes, 1))))
        file = open(o_path, 'wb')
        curvature_data.tofile(file)
        file.close()

    # Compute the ground truths out of multiple meshes
    if args.compute_all is not None:
        pass


