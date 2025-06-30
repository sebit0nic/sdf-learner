"""
File name: mesh.py
Author: Sebastian Lackner
Version: 1.0
Description: Computation of mesh curvature data
"""

import pymesh
import numpy as np
import argparse


def compute_curvature_points(in_file, out_file, debug):
    # Load mesh and convert to triangle mesh (pymesh only supports tri-meshes)
    mesh = pymesh.quad_to_tri(pymesh.load_mesh(in_file))

    # Compute gaussian curvature for all vertices
    mesh.add_attribute("vertex_gaussian_curvature")
    curvature = mesh.get_attribute("vertex_gaussian_curvature")

    # Store coordinates of vertices together absolute curvature (for sorting) and raw curvature (for visualization)
    curvature_data = np.hstack((mesh.vertices, np.reshape(np.absolute(curvature), (mesh.num_nodes, 1))))
    curvature_data = np.hstack((curvature_data, np.reshape(curvature, (mesh.num_nodes, 1))))
    file = open(out_file, 'wb')
    curvature_data.tofile(file)
    file.close()
    if debug:
        print(f'Wrote {mesh.num_nodes} curvatures to file.')


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
    out_file_extension = '.obj'
    curvature_folder = 'curvature/'
    curvature_file_prefix = 'sample'
    curvature_file_postfix = '_subdiv'
    curvature_file_extension = '.bin'

    # Compute the ground truth out of one mesh in the form [z, y, x, abs_curv, curv]
    if args.compute_one is not None:
        i_path = f'{in_folder}{in_file_prefix}{str(args.compute_one).zfill(6)}{in_file_postfix}{in_file_extension}'
        o_path = (f'{curvature_folder}{curvature_file_prefix}{str(args.compute_one).zfill(6)}'
                  f'{curvature_file_postfix}{curvature_file_extension}')
        compute_curvature_points(i_path, o_path, True)

    # Compute the ground truths out of multiple meshes in the form [z, y, x, abs_curv, curv]
    if args.compute_all is not None:
        for i in range(start_sample_num, sample_num):
            print(f'=> Computing sample {i + 1}')
            i_path = f'{in_folder}{in_file_prefix}{str(i).zfill(6)}{in_file_postfix}{in_file_extension}'
            o_path = (f'{curvature_folder}{curvature_file_prefix}{str(i).zfill(6)}{curvature_file_postfix}'
                      f'{curvature_file_extension}')
            compute_curvature_points(i_path, o_path, False)
