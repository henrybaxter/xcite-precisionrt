import argparse
from collections import namedtuple

import numpy as np


ESTEPE = 0  # this is a dummy value according to the manual

Phantom = namedtuple('Phantom', ['medias', 'boundaries', 'indices', 'densities'])


def make_phantom_cylinder(length, radius, voxel):
    # two layers of voxels that are not air
    x_min = -length / 2
    x_max = length / 2
    y_min = -radius
    y_max = radius
    z_min = 0
    z_max = 2 * radius
    n_x = int(np.ceil((x_max - x_min) / voxel))
    n_y = int(np.ceil((y_max - y_min) / voxel))
    n_z = int(np.ceil((z_max - z_min) / voxel))
    x_boundaries = np.linspace(x_min, x_max, n_x + 1)
    y_boundaries = np.linspace(y_min, y_max, n_y + 1)
    z_boundaries = np.linspace(z_min, z_max, n_z + 1)
    x_centers = (x_boundaries[1:] + x_boundaries[:-1]) / 2
    y_centers = (y_boundaries[1:] + y_boundaries[:-1]) / 2
    z_centers = (z_boundaries[1:] + z_boundaries[:-1]) / 2
    xx, yy, zz = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    # we remove one voxel on either side as the 'air'
    r2 = np.square(radius - voxel)
    in_cylinder = np.square(yy) + np.square(zz - radius) <= r2
    # default to 1 (air)
    indices = np.ones((n_x, n_y, n_z), dtype=np.int32)
    densities = np.full((n_x, n_y, n_z), 1.24e-3, dtype=np.float32)
    # set to 2 (water) if inside cylinder
    indices[in_cylinder] = 2
    densities[in_cylinder] = 1
    medias = ['Air_516kV', 'ICRUTISSUE516']
    boundaries = [x_boundaries, y_boundaries, z_boundaries]
    return Phantom(medias, boundaries, indices, densities)


def read_egsphant(fp):
    n_medias = int(fp.readline())
    medias = []
    for i in range(n_medias):
        medias.append(fp.readline().strip())
    for i in range(n_medias):
        fp.readline()  # skip ESTEPE
    shape = np.fromstring(fp.readline(), np.int32, sep=' ')
    boundaries = [np.fromstring(fp.readline(), np.float32, sep=' ') for i in range(3)]
    indices = []
    for z in range(shape[2]):
        fp.readline()  # skip a line
        slices = []
        for y in range(shape[1]):
            slices.append(np.array(list(map(int, fp.readline().strip()))))
        indices.append(slices)
    densities = []
    for z in range(shape[2]):
        fp.readline()
        slices = []
        for y in range(shape[1]):
            slices.append(np.fromstring(fp.readline(), np.float32, sep=' '))
        densities.append(slices)
    indices = np.array(indices).reshape(shape[::-1]).swapaxes(0, 2)
    densities = np.array(densities).reshape(shape[::-1]).swapaxes(0, 2)
    return Phantom(medias, boundaries, indices, densities)


def write_egsphant(phantom, fp):
    fp.write(' ' + str(len(phantom.medias)) + '\n')
    for media in phantom.medias:
        fp.write(media + '\n')
    for media in phantom.medias:
        fp.write('  {:.6f}'.format(ESTEPE) + '\n')
    fp.write('  ' + ' '.join(str(n) for n in phantom.indices.shape) + '\n')
    for boundary in phantom.boundaries:
        fp.write(' '.join(['{:.6f}'.format(b) for b in boundary]) + '\n')
    # xy slices
    n_x, n_y, n_z = [b.size - 1 for b in phantom.boundaries]
    for z in range(n_z):
        fp.write('\n')
        for y in range(n_y):
            out = ''.join(['{}'.format(v) for v in phantom.indices[:, y, z]])
            fp.write(out + '\n')
    for z in range(n_z):
        fp.write('\n')
        for y in range(n_y):
            out = ' '.join(['{:2E}'.format(v) for v in phantom.densities[:, y, z]])
            fp.write(out + '\n')
    fp.write('\n\n')


def create(args):
    print('Creating cylinder...')
    phantom = make_phantom_cylinder(args.length, args.radius, args.voxel)
    with open(args.output, 'w') as fp:
        write_egsphant(phantom, fp)
    print('Wrote cylinder to {}'.format(args.output))


def copy(args):
    print('Copying {} to {}'.format(args.input, args.output))
    with open(args.input) as ifp:
        with open(args.output, 'w') as ofp:
            write_egsphant(read_egsphant(ifp), ofp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    create_parser = subparsers.add_parser('create')
    create_parser.add_argument('output')
    create_parser.add_argument('--length', '-l', type=float, default=40.0)
    create_parser.add_argument('--radius', '-r', type=float, default=10.0)
    create_parser.add_argument('--voxel', type=float, default=1.0)
    create_parser.set_defaults(func=create)
    copy_parser = subparsers.add_parser('copy')
    copy_parser.add_argument('input')
    copy_parser.add_argument('output')
    copy_parser.set_defaults(func=copy)
    args = parser.parse_args()
    args.func(args)
