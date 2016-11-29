import argparse
import filecmp
from itertools import islice
from collections import namedtuple

import numpy

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


def read_3ddose(path):
    with open(path) as f:
        values = iter_values(f)
        # Row/Block 1 — number of voxels in x,y,z directions (e.g., nx, ny, nz)
        shape = numpy.fromiter(values, numpy.int32, 3)
        size = numpy.prod(shape)
        # Row/Block 2 — voxel boundaries (cm) in x direction(nx +1 values)
        # Row/Block 3 — voxel boundaries (cm) in y direction (ny +1 values)
        # Row/Block 4 — voxel boundaries (cm) in z direction(nz +1 values)
        boundaries = [numpy.fromiter(values, numpy.float32, n + 1) for n in shape]
        # print(boundaries)
        # Row/Block 5 — dose values array (nxnynz values)
        doses = numpy.fromiter(values, numpy.float32, size).reshape(shape)
        # print(doses)
        # Row/Block 6 — error values array (relative errors, nxnynz values)
        errors = numpy.fromiter(values, numpy.float32, size).reshape(shape)
        # print(errors)
        return Dose(boundaries, doses, errors)


def write_3ddose(path, dose):
    with open(path, 'w') as f:
        # Row/Block 1 — number of voxels in x,y,z directions (e.g., nx, ny, nz)
        write_lines(f, [len(boundary) - 1 for boundary in dose.boundaries], 'integer')
        # Row/Block 2 — voxel boundaries (cm) in x direction(nx +1 values)
        # Row/Block 3 — voxel boundaries (cm) in y direction (ny +1 values)
        # Row/Block 4 — voxel boundaries (cm) in z direction(nz +1 values)
        for boundary in dose.boundaries:
            write_lines(f, boundary, 'float')
        # Row/Block 5 — dose values array (nxnynz values)
        write_lines(f, numpy.nditer(dose.doses), 'float')
        # Row/Block 6 — error values array (relative errors, nxnynz values)
        write_lines(f, numpy.nditer(dose.errors), 'scientific')

ESTEPE = 0


# egsphant? read it in and dose it so
def read_egsphant(path):
    with open(path) as f:
        values = iter_values(f)
        medium_types = list(islice(values, int(next(values))))
        estepes = islice(values, len(medium_types))  # skip ESTEPE dummy values
        for estepe in estepes:
            assert int(float(estepe)) == ESTEPE
        shape = numpy.fromiter(values, numpy.int32, 3)
        size = numpy.prod(shape)
        boundaries = [numpy.fromiter(values, numpy.float32, n + 1) for n in shape]

        def medium_indices(values, shape):
            # just read it in and reshape it
            # read in shape[0] * shape[1] lines
            n_lines = int(shape[0] * shape[1])
            for xline in islice(values, n_lines):
                for index in xline:
                    yield index
        medium_indices = numpy.fromiter(medium_indices(values, shape), numpy.int32, size).reshape(shape)
        densities = numpy.fromiter(values, numpy.float32, size).reshape(shape)
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
    for line in f:
        for value in line.split():
            yield value


def apply_dose(phantom):
    # for now assume dosing of 1 and sphere of size 5 starting at the origin
    radius = 5
    origin = numpy.array([0, 0, 0])
    doses = numpy.zeros(phantom.medium_indices.shape)
    errors = numpy.zeros(phantom.medium_indices.shape)
    errors.fill(0.1)
    for x in range(len(phantom.boundaries[0]) - 1):
        for y in range(len(phantom.boundaries[1]) - 1):
            for z in range(len(phantom.boundaries[2]) - 1):
                corner = numpy.array([phantom.boundaries[0][x], phantom.boundaries[1][y], phantom.boundaries[2][z]])
                norm = numpy.linalg.norm(corner - origin)
                if norm < radius:
                    # drop off is exponential and zero after radius
                    doses[x, y, z] = 1 - numpy.power(norm / radius, 10)
    return Dose(phantom.boundaries, doses, errors)


def combine_3ddose(paths, output_path):
    doses = []
    errors = []
    for path in paths:
        dose = read_3ddose(path)
        boundaries = dose.boundaries
        doses.append(dose.doses)
        errors.append(dose.errors)
    doses = numpy.array(doses).sum(axis=0)
    errors = numpy.array(errors)
    errors = numpy.sqrt(numpy.square(errors).sum(axis=0))
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
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    if args.combine:
        combine_3ddose(args.input, args.combine)
    elif args.describe:
        assert len(args.input) == 1
        path = args.input[0]
        if path.endswith('.3ddose'):
            dose = read_3ddose(args.input[0])
        elif path.endswith('.egsphant'):
            phantom = read_egsphant(args.input[0])
            for axis, label in enumerate(['x', 'y', 'z']):
                print('{}'.format(label), phantom.boundaries[axis][-1])
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
        phantom = read_egsphant(args.input)
        dose = apply_dose(phantom)
        write_3ddose(args.dose, dose)
    else:
        dose1 = read_3ddose(args.input)
        dose2 = read_3ddose(args.input)
        result = dose2.doses - dose1.doses
        print(result)
