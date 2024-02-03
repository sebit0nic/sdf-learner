from SDFFileHandler import SDFReader
from SDFFileHandler import SDFWriter
from SDFUtil import SDFCurvature
from SDFVisualizer import SDFVisualizer
import argparse
import time
import torch


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
                        help='Export graphics as pyvista graph.')
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

    # in_tensor = torch.as_tensor(samples, device=torch.device('cuda'), dtype=float)
    # print(in_tensor)
