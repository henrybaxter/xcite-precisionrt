import os
import argparse
import logging
import filecmp
from itertools import islice, chain
from collections import namedtuple
from functools import reduce, partial

import numpy as np
import scipy.optimize as optimize
from natsort import natsorted

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

logger = logging.getLogger(__name__)


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


def target_to_skin(dose, target):
    # skin based on first medium density change?
    skin_indices = [[2] * 25, list(range(48, 53)) * 5, list(range(48, 53)) * 5]
    skin_mean = np.sum(dose.doses[skin_indices]) / 25
    logger.info('Found skin mean {}'.format(skin_mean))

    centers = [(b[1:] + b[:-1]) / 2 for b in dose.boundaries]
    translated = centers - target.isocenter[:, np.newaxis]
    d2 = reduce(np.add.outer, np.square(translated))
    r2 = np.square(target.radius)
    target_indices = np.where(d2 < r2)
    target_mean = np.sum(dose.doses[target_indices]) / len(target_indices[0])
    logger.info('Found target mean {}'.format(target_mean))

    result = target_mean / skin_mean
    logger.info('So target to skin dose is {}'.format(result))
    return result


def optimize_stt(paths, target, output):
    logger.info('Reading raw doses')
    boundaries = read_3ddose(paths[0]).boundaries
    fname = 'unweighted.npy'
    if os.path.exists(fname):
        logger.info('Found cached unweighted')
        unweighted = np.load(fname)
    else:
        logger.info('No cached version found')
        unweighted = np.array([read_3ddose(path).doses.ravel() for path in paths])
        logger.info('Saving to cache')
        np.save('unweighted.npy', unweighted)
    print('unweighted shape is', unweighted.shape)
    unweighted /= np.max(unweighted)
    logger.info('Generating indices')
    # ok what if we only grabbed those indices, and then weighted them?
    skin_indices = np.vstack(map(np.ravel, np.mgrid[2:4, 40:60, 40:60]))
    print('skin indices shape', skin_indices)
    skin_indices = np.ravel_multi_index(skin_indices, dims=[100, 100, 100])
    print('skin indices shape', skin_indices.shape)

    centers = [(b[1:] + b[:-1]) / 2 for b in boundaries]
    translated = centers - target.isocenter[:, np.newaxis]
    d2 = reduce(np.add.outer, np.square(translated))
    r2 = np.square(target.radius)
    target_indices = np.where(d2 < r2)
    print('target indices', target_indices)
    target_indices = np.ravel_multi_index(target_indices, dims=[100, 100, 100])
    print('target indices shape', target_indices.shape)

    # initial_weights = np.array(list(range(0, len(paths))))
    logger.info('Starting tensorflow')
    import tensorflow as tf
    sess = tf.InteractiveSession()
    skin_values = unweighted[:, skin_indices].T
    target_values = unweighted[:, target_indices].T
    coeffs = np.polyfit([0, len(paths) // 2, len(paths) - 1], [4, 1, 4], 2)
    initial_weights = np.polyval(coeffs, np.arange(0, len(paths)))
    X_skin = tf.constant(skin_values, name='skin_doses')
    X_target = tf.constant(target_values, name='target_doses')
    W = tf.Variable(initial_weights[:, np.newaxis], name='weights', dtype=tf.float32)
    skin = tf.matmul(X_skin, W)
    target = tf.matmul(X_target, W)
    loss = s_to_t = tf.reduce_mean(skin) / tf.reduce_mean(target)
    mean, variance = tf.nn.moments(skin, axes=[0])
    no_neg = -tf.minimum(tf.reduce_min(W) - 1, 0) * 1000
    no_big = tf.minimum(tf.reduce_max(W) - 15, 0) * 1000
    reg = no_neg + no_big
    train_step = tf.train.AdamOptimizer().minimize(loss + reg)
    sess.run(tf.global_variables_initializer())
    steps = 0
    for i in range(steps):
        _, w, st = sess.run([train_step, W, s_to_t])
        print('Current skin to target is {}'.format(st))
    w = sess.run(W)
    final_weights = w.flatten()
    # generate a weighted 3ddose file
    weight_3ddose(paths, output, final_weights)


    print(list(final_weights))
    logger.info('Have final weights, now applying them')

    logger.info('Done')


def optimize_stt_(paths, target):
    # assumes doses is a lise of of dose.doses
    # bounds at 1 to 50x weighting
    logger.info("Loading {} dose files for weight optimization".format(len(paths)))
    boundaries = read_3ddose(paths[0]).boundaries
    doses = np.array([read_3ddose(path).doses for path in paths])
    logger.info("Doses loaded")
    bounds = [(1, 50)] * len(doses)
    # i think 1 in the center to 15 on the outside
    # following a parabola (3 constraints)
    coeffs = np.polyfit([0, len(doses) // 2, len(doses)], [15, 1, 15], 2)
    initial_weights = np.polyval(coeffs, np.arange(0, len(doses)))
    logger.info('Initial guess weights are:\n{}'.format(initial_weights))

    def objective(weights):
        logger.info("Asked to score weights\n{}".format(initial_weights - weights))
        dose = Dose(boundaries, (doses.T * weights).T.sum(axis=0), None)
        score = stt(target, dose)
        logger.info("Score is now {}".format(score))
        return score

    def in_bounds(**kwargs):
        is_in = bool(np.all(kwargs['x_new'] >= 1))
        logger.info('In bounds? {}'.format(is_in))
        return is_in

    def take_step(x):
        print(x)
        x += np.random.uniform(low=-0.5, high=.5, size=[len(doses)])
        print(x)
        return x
    result = optimize.basinhopping(objective, initial_weights, take_step=take_step) # , bounds=bounds, options={'disp': True, 'eps': 1e0})
    logger.info('Resulting weights are:\n{}'.format(result))


# what about total dose to skin over total dose to target?
# or what about maximum normalized skin dose to maximum target dose?

def dose_to_grays(dose, minutes=30, milliamps=200):
    # what about number of electrons though?
    seconds = float(minutes) * 60
    amps = float(milliamps) / 1000
    return dose * amps * seconds / (1.602176 * np.power(10, -19.0))


def dvh(dose, target):
    """
    Assumptions:

        - dose.boundaries is x, y, z
        - dose.doses is z, y, x
        - target.isocenter is x, y, z
        - target.radius
    """
    # temporary hack for bad 3ddose read/writing, we swap x and z to fix it
    doses = np.swapaxes(dose.doses, 0, 2)
    centers = [(b[1:] + b[:-1]) / 2 for b in dose.boundaries]
    translated = centers - target.isocenter[:, np.newaxis]
    v = volumes(dose.boundaries)
    d2 = reduce(np.add.outer, np.square(translated))
    r2 = np.square(target.radius)
    in_target = d2 < r2
    # target_volume = np.sum(v[np.where(in_target)])
    print(dose_to_grays(np.mean(doses[np.where(in_target)])) / (74 * 24))
    print(dose_to_grays(np.min(doses[np.where(in_target)])) / (74 * 24))
    print(dose_to_grays(np.max(doses[np.where(in_target)])) / (74 * 24))

    # so we take the minimum? the max? take the max
    BINS = 1000
    # more than 0 grays, 100% of volume
    # more than 1 grays, 99% of volume...
    # ok so then we find
    # more than 0 grays
    # more than 1 grays
    # more than 2 grays..
    max_dose = np.max(doses)
    dose_increment = max_dose / (BINS - 1)
    target_volume = np.sum(v[np.where(in_target)])
    result = []
    for i in range(BINS):
        current = i * dose_increment
        greater_than_current = doses > current
        should_count = np.logical_and(in_target, greater_than_current)
        percent_vol = np.sum(v[np.where(should_count)]) / target_volume
        print(current, percent_vol)
        result.append((dose_to_grays(current) / (74 * 24), percent_vol))
    return result


def make_phantom_cylinder(length, radius, voxel):
    # two layers of voxels that are not air
    y_max = length
    y_min = 0
    x_max = radius
    x_min = -x_max
    z_max = radius
    z_min = -z_max
    output = []
    media_types = ['Air_516kV', 'ICRUTISSUE516']
    media_densities = ['1.240000e-03', '1']
    output.append(' {}'.format(len(media_types)))
    for media in media_types:
        output.append(media)
    for media in media_types:
        output.append('  0.000000')
    n_x = int(np.ceil((x_max - x_min) / voxel))
    n_y = int(np.ceil((y_max - y_min) / voxel))
    n_z = int(np.ceil((z_max - z_min) / voxel))
    print(n_x, n_y, n_z)
    output.append('  {} {} {}'.format(n_x, n_y, n_z))
    def ok(f):
        return '{:.6f}'.format(f)
    x_boundaries = np.linspace(x_min, x_max, n_x + 1)
    output.append('  '.join(map(ok, x_boundaries)))
    y_boundaries = np.linspace(y_min, y_max, n_y + 1)
    output.append('  '.join(map(ok, y_boundaries)))
    z_boundaries = np.linspace(z_min, z_max, n_z + 1)
    output.append('  '.join(map(ok, z_boundaries)))
    # ok now we check to see if it's in the cylinder.
    # we assume the cylinder stretches the whole length, but we miss two voxels on either side (or .8mm?)
    # no, two voxels.
    x_centers = (x_boundaries[1:] + x_boundaries[:-1]) / 2
    y_centers = (y_boundaries[1:] + y_boundaries[:-1]) / 2
    z_centers = (z_boundaries[1:] + z_boundaries[:-1]) / 2
    xx, yy, zz = np.meshgrid(x_centers, y_centers, z_centers)
    # ok now we need to average
    r2 = np.square(radius - voxel)
    # now we need to find anything inside the cylinder, and we assume anything along y is, so
    # x = np.square(x_centers) <= r2h
    # z = np.square(z_centers) <= r2h
    in_cylinder = np.square(xx) + np.square(zz) <= r2
    mediums = np.ones((n_x, n_y, n_z), dtype=np.int32)
    mediums[in_cylinder] = 2
    # print(mediums)
    output.append('')
    for z in range(n_z):
        for x in range(n_x):
            output.append(''.join(map(str, mediums[x, :, z])))
        output.append('')
    for z in range(n_z):
        for x in range(n_x):
            densities = [media_densities[i-1] for i in mediums[x, :, z]]
            output.append(' '.join(densities))
        output.append('')
    output.append('')
    output.append('')
    open('test.egsphant', 'w').write("\n".join(output))


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
    logger.info('Target hit {}'.format(underdosed))
    logger.info('Tissue avoided {}'.format(overdosed))
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


def write_npz(path, dose):
    np.savez_compressed(path, **dose._asdict())


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
    weights = np.array(weights)
    assert list(weights.shape) == [len(paths)], '{} != {}'.format(list(weights.shape), [len(paths)])
    doses = []
    # errors = []
    for path, weight in zip(paths, weights):
        # print('Weighting {} at {}'.format(path, weight))
        dose = read_3ddose(path)
        boundaries = dose.boundaries
        doses.append(dose.doses)
        errors = dose.errors
        # errors.append(dose.errors)
    doses = (np.array(doses).T * weights).T.sum(axis=0)
    print('Doses calculated')
    print('Max is {}'.format(np.amax(doses)))
    # doses = np.tensordot(doses, weights, axes=1)
    # errors = np.array(errors)
    # errors = np.sqrt(np.square(errors).sum(axis=0))
    write_3ddose(output_path, Dose(boundaries, doses, errors))


def normalize_3ddose(path, output_path):
    dose = read_3ddose(path)
    result = dose.doses / dose.doses.sum()
    write_3ddose(output_path, Dose(dose.boundaries, result, dose.errors))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+')
    parser.add_argument('--dvh', action='store_true')
    parser.add_argument('--phantom', action='store_true')
    parser.add_argument('--test-3ddose', action='store_true')
    parser.add_argument('--test-egsphant', action='store_true')
    parser.add_argument('--dose')
    parser.add_argument('--combine')
    parser.add_argument('--normalize')
    parser.add_argument('--weight')
    parser.add_argument('--errors', action='store_true')
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--compress', action='store_true')
    parser.add_argument('--decompress', action='store_true')
    parser.add_argument('--paddick', action='store_true')
    parser.add_argument('--optimize-stt', action='store_true')
    args = parser.parse_args()
    args.input = natsorted(args.input)
    if args.phantom:
        make_phantom_cylinder(20, 10, .1)
    elif args.dvh:
        dose = read_3ddose(args.input[0])
        target = Target(np.array([0, 20, -10]), 1)
        dvh(dose, target)
    elif args.compress:
        read_3ddose(args.input[0])
        os.remove(args.input[0])
    elif args.decompress:
        for path in args.input:
            dose = read_3ddose(path)
            write_3ddose(path.replace('.npz', ''), dose)
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
        print('skin to target ratio', stt(dose, target))
    elif args.optimize_stt:
        target_origin = np.array([-10, 20, 0])
        target_radius = 1
        target = Target(target_origin, target_radius)
        optimize_stt(args.input, target, args.optimize_stt)
    else:
        dose1 = read_3ddose(args.input)
        dose2 = read_3ddose(args.input)
        result = dose2.doses - dose1.doses
        print(result)
