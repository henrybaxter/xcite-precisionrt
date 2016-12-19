import os
import pickle
import argparse
import sys
import filecmp
from itertools import islice, chain
from collections import namedtuple
from functools import reduce

import numpy as np

"""
http://nrc-cnrc.github.io/EGSnrc/doc/pirs794-dosxyznrc.pdf

Row/Block 1 — number of voxels in x,y,z directions (e.g., nx, ny, nz)
Row/Block 2 — voxel boundaries (cm) in x direction(nx +1 values)
Row/Block 3 — voxel boundaries (cm) in y direction (ny +1 values)
Row/Block 4 — voxel boundaries (cm) in z direction(nz +1 values)
Row/Block 5 — dose values array (nxnynz values)
Row/Block 6 — error values array (relative errors, nxnynz values)
"""


Dose = namedtuple('Dose', ['boundaries', 'doses', 'errors'])
Phantom = namedtuple('Phantom', ['medium_types', 'boundaries', 'medium_indices', 'densities'])
Target = namedtuple('Target', ['isocenter', 'radius'])


def volumes(boundaries):
    """Given a 3-tuple of x, y, z, returns an array of len(x) x len(y) x len(z)
    where each [x][y][z] is the volume at that index.
    >>> volumes([np.array([1, 2, 3]), np.array([2, 4, 6, 8]), np.array([3, 6, 9, 12, 15])])
    array([[[6, 6, 6, 6],
            [6, 6, 6, 6],
            [6, 6, 6, 6]],
    <BLANKLINE>
           [[6, 6, 6, 6],
            [6, 6, 6, 6],
            [6, 6, 6, 6]]])
    """
    return reduce(np.multiply.outer, [b[1:] - b[:-1] for b in boundaries])


def simplified_skin_to_target_ratio(dose, target):
    # skin based on first medium density change?
    skin_indices = [[2] * 100, list(range(45, 55)) * 10, list(range(45, 55)) * 10]
    skin_mean = np.sum(dose.doses[skin_indices]) / 100
    print('skin mean', skin_mean)

    centers = [(b[1:] + b[:-1]) / 2 for b in dose.boundaries]
    translated = centers - target.isocenter[:, np.newaxis]
    d2 = reduce(np.add.outer, np.square(translated))
    r2 = np.square(target.radius)
    target_indices = np.where(d2 < r2)
    target_mean = np.sum(dose.doses[target_indices]) / len(target_indices[0])
    print('target mean', target_mean)

    return skin_mean / target_mean

# what about total dose to skin over total dose to target?
# or what about maximum normalized skin dose to maximum target dose?


def paddick(dose, target):
    centers = [(b[1:] + b[:-1]) / 2 for b in dose.boundaries]
    translated = centers - target.isocenter[:, np.newaxis]
    index = np.argmin(np.abs(translated), axis=1)
    reference_dose = dose.doses[tuple(index)]
    normalized = dose.doses / reference_dose
    # get target volume
    # by finding indices of all points in a volume and finding their volume
    v = volumes(dose.boundaries)
    d2 = reduce(np.add.outer, np.square(translated))
    r2 = np.square(target.radius)
    in_target = d2 < r2
    in_dosed = normalized >= reference_dose * .8
    in_both = np.logical_and(in_target, in_dosed)
    target_volume = np.sum(v[np.where(in_target)])
    dosed_volume = np.sum(v[np.where(in_dosed)])
    both_volume = np.sum(v[np.where(in_both)])
    # print('target volume', target_volume)
    # print('dosed volume', dosed_volume)
    # print('both volume', both_volume)
    underdosed = both_volume / target_volume
    overdosed = both_volume / dosed_volume
    # higher is better
    print('target hit', underdosed)
    print('tissue avoided', overdosed)
    return underdosed * overdosed


def _read_3ddose(path):
    with open(path) as f:
        values = iter_values(f)
        # Row/Block 1 — number of voxels in x,y,z directions (e.g., nx, ny, nz)
        shape = np.fromiter(values, np.int32, 3)  # shape in x, y, z
        # Row/Block 2 — voxel boundaries (cm) in x direction(nx +1 values)
        # Row/Block 3 — voxel boundaries (cm) in y direction (ny +1 values)
        # Row/Block 4 — voxel boundaries (cm) in z direction(nz +1 values)
        boundaries = np.flipud([np.fromiter(values, np.float32, n + 1) for n in shape])
        shape = np.flipud(shape)
        size = np.prod(shape)
        # print(boundaries)
        # Row/Block 5 — dose values array (nxnynz values)
        doses = np.fromiter(values, np.float32, size).reshape(shape)
        # print(doses)
        # Row/Block 6 — error values array (relative errors, nxnynz values)
        errors = np.fromiter(values, np.float32, size).reshape(shape)
        # print(errors)
        return Dose(boundaries, doses, errors)


def read_3ddose(path):
    if path.endswith('.npz'):
        path = ''.join(path.rsplit('.npz', 1))
    npz_path = path + '.npz'
    if not os.path.exists(npz_path):
        np.savez_compressed(npz_path, **_read_3ddose(path)._asdict())
    return Dose(**np.load(npz_path))


def write_3ddose(path, dose):
    assert len(dose.doses.shape) == 3, "Doses must be 3d array"
    assert len(dose.errors.shape) == 3, "Errors must be 3d array"
    with open(path, 'w') as f:
        boundaries = np.flipud(dose.boundaries)
        # Row/Block 1 — number of voxels in x,y,z directions (e.g., nx, ny, nz)
        write_lines(f, [len(boundary) - 1 for boundary in boundaries], 'integer')
        # Row/Block 2 — voxel boundaries (cm) in x direction(nx +1 values)
        # Row/Block 3 — voxel boundaries (cm) in y direction (ny +1 values)
        # Row/Block 4 — voxel boundaries (cm) in z direction(nz +1 values)
        for boundary in boundaries:
            write_lines(f, boundary, 'float')
        # Row/Block 5 — dose values array (nxnynz values)
        write_lines(f, np.nditer(dose.doses), 'scientific')
        # Row/Block 6 — error values array (relative errors, nxnynz values)
        write_lines(f, np.nditer(dose.errors), 'scientific')

ESTEPE = 0


# egsphant? read it in and dose it so
def read_egsphant(path):
    with open(path) as f:
        values = iter_values(f)
        medium_types = list(islice(values, int(next(values))))
        estepes = islice(values, len(medium_types))  # skip ESTEPE dummy values
        for estepe in estepes:
            assert int(float(estepe)) == ESTEPE
        shape = np.fromiter(values, np.int32, 3)
        size = np.prod(shape)
        boundaries = [np.fromiter(values, np.float32, n + 1) for n in shape]

        def medium_indices(values, shape):
            # just read it in and reshape it
            # read in shape[0] * shape[1] lines
            n_lines = int(shape[0] * shape[1])
            for xline in islice(values, n_lines):
                for index in xline:
                    yield index
        medium_indices = np.fromiter(medium_indices(values, shape), np.int32, size).reshape(shape)
        densities = np.fromiter(values, np.float32, size).reshape(shape)
        return Phantom(medium_types, boundaries, medium_indices, densities)


def write_egsphant(path, phantom):
    with open(path, 'w') as f:
        f.write(str(len(phantom.medium_types)) + '\n')
        for medium in phantom.medium_types:
            f.write(medium + '\n')
        f.write(' '.join([ESTEPE for i in range(len(phantom.medium_types))]) + '\n')
        f.write(' '.join([str(len(boundary) - 1) for boundary in phantom.boundaries]) + '\n')
        for boundary in phantom.boundaries:
            f.write(' '.join(['{:.6f}'.format(b) for b in boundary]) + '\n')
        for zslice in phantom.medium_indices:
            for y in zslice:
                f.write(''.join([str(x) for x in y]) + '\n')


def write_lines(f, values, typ=None):
    values = iter(values)
    if typ == 'integer':
        def fmt(v):
            return str(v)
    elif typ == 'float':
        def fmt(v):
            v = float(v)
            s = '{:.4f}'.format(v)
            if v < 0:
                return s[:7]
            else:
                return s[:6]
    elif typ == 'scientific':
        def fmt(v):
            return '{:.4E}'.format(float(v))
    else:
        def fmt(v):
            return str(v)

    while True:
        to_write = [fmt(v) for v in islice(values, 5)]
        if not to_write:
            break
        f.write(' '.join(to_write) + '\n')


def iter_values(f):
    return chain(*[line.split() for line in f])


def apply_dose(phantom):
    # for now assume dosing of 1 and sphere of size 5 starting at the origin
    radius = 2
    radius_2 = radius * radius
    origin = np.array([-10, 20, 0])
    doses = np.zeros(phantom.medium_indices.shape)
    errors = np.zeros(phantom.medium_indices.shape)
    errors.fill(0.1)
    for x in range(len(phantom.boundaries[0]) - 1):
        for y in range(len(phantom.boundaries[1]) - 1):
            for z in range(len(phantom.boundaries[2]) - 1):
                corner = np.array([phantom.boundaries[0][x], phantom.boundaries[1][y], phantom.boundaries[2][z]])
                vector = corner - origin
                r_2 = np.square(vector).sum()
                if r_2 < radius_2:
                    doses[x, y, z] = 1
                    # drop off is exponential and zero after radius
                    # doses[x, y, z] = 1 - np.power(norm / radius, 10)
    return Dose(phantom.boundaries, doses, errors)


def combine_3ddose(paths, output_path):
    # possibly chunk this - sum in groups of 100 or something?
    doses = []
    # errors = []
    for path in paths:
        dose = read_3ddose(path)
        boundaries = dose.boundaries
        errors = dose.errors
        doses.append(dose.doses)
        # errors.append(dose.errors)
    doses = np.array(doses).sum(axis=0)
    # errors = np.array(errors)
    # errors = np.sqrt(np.square(errors).sum(axis=0))
    write_3ddose(output_path, Dose(boundaries, doses, errors))


def weight_3ddose(paths, output_path, weights):
    doses = []
    errors = []
    for path, weight in zip(paths, weights):
        print('Weighting {} at {}'.format(path, weight))
        dose = read_3ddose(path)
        boundaries = dose.boundaries
        doses.append(dose.doses)
        errors.append(dose.errors)
    doses = (np.array(doses).T * weights).T.sum(axis=0)
    print('Doses calculated')
    print('Max is {}'.format(np.amax(doses)))
    # doses = np.tensordot(doses, weights, axes=1)
    errors = np.array(errors)
    errors = np.sqrt(np.square(errors).sum(axis=0))
    write_3ddose(output_path, Dose(boundaries, doses, errors))


def normalize_3ddose(path, output_path):
    dose = read_3ddose(path)
    result = dose.doses / dose.doses.sum()
    write_3ddose(output_path, Dose(dose.boundaries, result, dose.errors))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+')
    parser.add_argument('--test-3ddose', action='store_true')
    parser.add_argument('--test-egsphant', action='store_true')
    parser.add_argument('--dose')
    parser.add_argument('--combine')
    parser.add_argument('--normalize')
    parser.add_argument('--weight')
    parser.add_argument('--errors', action='store_true')
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--compress', action='store_true')
    parser.add_argument('--uncompress', action='store_true')
    parser.add_argument('--paddick', action='store_true')
    args = parser.parse_args()
    if args.compress:
        read_3ddose(args.input[0])
        os.remove(args.input[0])
    elif args.uncompress:
        dose = read_3ddose(args.input[0])
        write_3ddose(args.input[0].replace('.npz', ''), dose)
    elif args.errors:
        dose = read_3ddose(args.input[0])
        print('{} unique error values'.format(np.unique(dose.errors).size))
    elif args.combine:
        combine_3ddose(args.input, args.combine)
    elif args.weight:
        weight_3ddose(args.input, args.weight, np.ones(len(args.input)))
    elif args.describe:
        assert len(args.input) == 1
        path = args.input[0]
        if path.endswith('.3ddose'):
            dose = read_3ddose(args.input[0])
        elif path.endswith('.egsphant'):
            phantom = read_egsphant(args.input[0])
            for axis, label in enumerate(['x', 'y', 'z']):
                print('{} [{}, {}]'.format(label, phantom.boundaries[axis][0], phantom.boundaries[axis][-1]))
        else:
            raise ValueError("Cannot describe {} files".format(path.split('.')[1]))
    elif args.normalize:
        assert len(args.input) == 1
        normalize_3ddose(args.input[0], args.normalize)
    elif args.test_3ddose:
        test_path = args.input + '.test'
        write_3ddose(test_path, read_3ddose(args.input))
        if not filecmp.cmp(args.input, test_path):
            print('Files {} and {} differ'.format(args.input, test_path))
    elif args.test_egsphant:
        test_path = args.input + '.test'
        write_egsphant(test_path, read_egsphant(args.input))
        if not filecmp.cmp(args.input, test_path):
            print('Files {} and {} differ'.format(args.input, test_path))
    elif args.dose:
        phantom = read_egsphant(args.input[0])
        dose = apply_dose(phantom)
        write_3ddose(args.dose, dose)
    elif args.paddick:
        dose = read_3ddose(args.input[0])
        # target is in x, y, z coordinates
        target_origin = np.array([-10, 20, 0])
        target_radius = 2
        target = Target(target_origin, target_radius)
        print('paddick', paddick(dose, target))
        print('skin to target ratio', simplified_skin_to_target_ratio(dose, target))
    else:
        dose1 = read_3ddose(args.input)
        dose2 = read_3ddose(args.input)
        result = dose2.doses - dose1.doses
        print(result)
