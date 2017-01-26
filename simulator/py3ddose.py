import os
import argparse
import logging
from collections import namedtuple
from functools import reduce
from zipfile import BadZipFile

import numpy as np
import scipy.optimize as optimize

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
    # if we knew the area of skin affected?
    # how do we choose it? how about we use 2cmx2cmx4mm
    skin_indices = [[2] * 25, list(range(48, 53)) * 5, list(range(48, 53)) * 5]
    skin_mean = np.sum(dose.doses[skin_indices]) / 25
    logger.info('Found skin mean {}'.format(skin_mean))

    centers = [(b[1:] + b[:-1]) / 2 for b in dose.boundaries]
    translated = [c - target.isocenter[i] for i, c in enumerate(centers)]
    xx, yy, zz = np.meshgrid(*translated, indexing='ij')
    d2 = np.square(xx) + np.square(yy) + np.square(zz)
    r2 = np.square(target.radius)
    target_indices = np.where(d2 < r2)
    target_mean = np.sum(dose.doses[target_indices]) / len(target_indices[0])
    logger.info('Found target mean {}'.format(target_mean))

    result = target_mean / skin_mean
    logger.info('So target to skin dose is {}'.format(result))
    return result


def reflect(dose):
    # reflect the dose around an axis
    # that means we need to have a symmetric guy
    # is that realistic? i'm not sure
    # plus we could just do a quarter of it
    # and then re-weight
    # no let's not
    pass


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


def dose_stats(dose, target):
    # length of 3 - x centers, y centers, z centers
    centers = [(b[1:] + b[:-1]) / 2 for b in dose.boundaries]
    translated = [c - target.isocenter[i] for i, c in enumerate(centers)]
    xx, yy, zz = np.meshgrid(*translated, indexing='ij')
    d2 = np.square(xx) + np.square(yy) + np.square(zz)
    r2 = np.square(target.radius)
    in_target = d2 < r2
    # minimum dose to 90% of the
    # find all, flatten, sort, then find top 90%, etc
    sorted_doses = np.sort(dose.doses[np.where(in_target)].reshape(-1))
    absolute = {
        'max': np.max(dose.doses[np.where(in_target)]),
        'min': np.min(dose.doses[np.where(in_target)]),
        'mean': np.mean(dose.doses[np.where(in_target)]),
        '90': np.min(sorted_doses[-int(sorted_doses.size*.9):]),
        '95': np.min(sorted_doses[-int(sorted_doses.size*.95):]),
        '100': np.min(dose.doses[np.where(in_target)])
    }
    percent = {}
    for key, value in absolute.items():
        percent[key] = absolute[key] / absolute['max']
    for key, value in absolute.items():
        absolute[key] = dose_to_grays(absolute[key]) / (74 * 24)
    return {
        'percent': percent,
        'absolute': absolute
    }


def dvh(dose, target):
    """
    Assumptions:

        - dose.boundaries is x, y, z
        - dose.doses is z, y, x
        - target.isocenter is x, y, z
        - target.radius
    """
    # temporary hack for bad 3ddose read/writing, we swap x and z to fix it
    centers = [(b[1:] + b[:-1]) / 2 for b in dose.boundaries]
    translated = [c - target.isocenter[i] for i, c in enumerate(centers)]
    xx, yy, zz = np.meshgrid(*translated, indexing='ij')
    d2 = np.square(xx) + np.square(yy) + np.square(zz)
    v = volumes(dose.boundaries)
    r2 = np.square(target.radius)
    in_target = d2 < r2
    # target_volume = np.sum(v[np.where(in_target)])
    # print(dose_to_grays(np.mean(dose.doses[np.where(in_target)])) / (74 * 24))
    # print(dose_to_grays(np.min(dose.doses[np.where(in_target)])) / (74 * 24))
    # print(dose_to_grays(np.max(dose.doses[np.where(in_target)])) / (74 * 24))

    # so we take the minimum? the max? take the max
    BINS = 100
    # more than 0 grays, 100% of volume
    # more than 1 grays, 99% of volume...
    # ok so then we find
    # more than 0 grays
    # more than 1 grays
    # more than 2 grays..
    max_dose = np.max(dose.doses)
    dose_increment = max_dose / (BINS - 1)
    target_volume = np.sum(v[np.where(in_target)])
    result = []
    for i in range(BINS):
        current = i * dose_increment
        greater_than_current = dose.doses > current
        should_count = np.logical_and(in_target, greater_than_current)
        percent_vol = np.sum(v[np.where(should_count)]) / target_volume
        # print(current, percent_vol)
        result.append((dose_to_grays(current), percent_vol))
    return result


def paddick(dose, target):
    centers = [(b[1:] + b[:-1]) / 2 for b in dose.boundaries]
    translated = [c - target.isocenter[i] for i, c in enumerate(centers)]
    xx, yy, zz = np.meshgrid(*translated, indexing='ij')
    # get target volume
    # by finding indices of all points in a volume and finding their volume
    v = volumes(dose.boundaries)
    d2 = np.square(xx) + np.square(yy) + np.square(zz)
    r2 = np.square(target.radius)
    in_target = d2 < r2
    in_dosed = dose.doses >= 1e-19
    in_both = np.logical_and(in_target, in_dosed)
    target_volume = np.sum(v[np.where(in_target)])
    print('target volume', target_volume)
    dosed_volume = np.sum(v[np.where(in_dosed)])
    print('dosed volume', dosed_volume)
    both_volume = np.sum(v[np.where(in_both)])
    # print('target volume', target_volume)
    # print('dosed volume', dosed_volume)
    print('both volume', both_volume)
    underdosed = both_volume / target_volume
    overdosed = both_volume / dosed_volume
    # higher is better
    logger.info('Target hit {}'.format(underdosed))
    logger.info('Tissue avoided {}'.format(overdosed))
    return underdosed * overdosed


def _read_3ddose(path):
    with open(path) as f:
        # Row/Block 1 — number of voxels in x,y,z directions (e.g., nx, ny, nz)
        shape = np.fromstring(f.readline(), np.int32, sep=' ')
        # Row/Block 2 — voxel boundaries (cm) in x direction(nx +1 values)
        # Row/Block 3 — voxel boundaries (cm) in y direction (ny +1 values)
        # Row/Block 4 — voxel boundaries (cm) in z direction(nz +1 values)
        boundaries = [np.fromstring(f.readline(), np.float32, sep=' ') for n in shape]
        # Row/Block 5 — dose values array (nxnynz values)
        doses = np.fromstring(f.readline(), np.float32, sep=' ')
        doses = doses.reshape(shape[::-1]).swapaxes(0, 2)
        # Row/Block 6 — error values array (relative errors, nxnynz values)
        errors = np.fromstring(f.readline(), np.float32, sep=' ').reshape(shape[::-1]).swapaxes(0, 2)
        return Dose(boundaries, doses, errors)


def read_3ddose(path):
    if path.endswith('.npz'):
        path = ''.join(path.rsplit('.npz', 1))
    npz_path = path + '.npz'
    if os.path.exists(npz_path):
        try:
            return Dose(**np.load(npz_path))
        except (BadZipFile, OSError):
            logger.error('File at {} is bad, removing'.format(npz_path))
            os.remove(npz_path)
    dose = _read_3ddose(path)
    write_npz(npz_path, dose)
    return dose


def write_npz(path, dose):
    np.savez_compressed(path, **dose._asdict())


def write_3ddose(path, dose):
    print('Writing {}'.format(path))
    assert len(dose.doses.shape) == 3, "Doses must be 3d array"
    assert len(dose.errors.shape) == 3, "Errors must be 3d array"
    with open(path, 'w') as f:
        boundaries = dose.boundaries

        # Row/Block 1 — number of voxels in x,y,z directions (e.g., nx, ny, nz)
        f.write(' '.join(map(str, np.array(dose.doses.shape))) + '\n')
        # Row/Block 2 — voxel boundaries (cm) in x direction(nx +1 values)
        # Row/Block 3 — voxel boundaries (cm) in y direction (ny +1 values)
        # Row/Block 4 — voxel boundaries (cm) in z direction(nz +1 values)
        for boundary in boundaries:
            f.write(' '.join(['{:.4f}'.format(v) for v in boundary]) + '\n')
        # Row/Block 5 — dose values array (nxnynz values)
        f.write(' '.join(['{:.4E}'.format(v) for v in dose.doses.swapaxes(0, 2).reshape(-1)]) + '\n')
        # Row/Block 6 — error values array (relative errors, nxnynz values)
        f.write(' '.join(['{:.16f}'.format(v) for v in dose.errors.swapaxes(0, 2).reshape(-1)]) + '\n')


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
    # need to normalize the weights, so that the sum is 1
    # which means taking the sum of the weights and dividing by that
    weights = np.array(weights)
    weights /= np.sum(weights)
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
    parser.add_argument('--decompress', '-x', action='store_true')
    parser.add_argument('--stats', action='store_true')
    parser.add_argument('--output', '-o')
    parser.add_argument('--max', action='store_true')
    parser.add_argument('--sum', action='store_true')
    args = parser.parse_args()
    if args.max:
        for inp in args.input:
            dose = read_3ddose(inp)
            print(np.max(dose.doses))
    elif args.sum:
        for inp in args.input:
            dose = read_3ddose(inp)
            print(np.sum(dose.doses))
    elif args.decompress:
        for inp in args.input:
            dose = read_3ddose(inp)
            write_3ddose(inp.replace('.npz', ''), dose)
    #write_3ddose(args.output, read_3ddose(args.input))
    #target = Target(np.array([0, 10, -10]), 4)
    #dose = read_3ddose('reports/Stamped-1-row-0.2mm-Septa/dose/arc.3ddose')
    elif args.stats:
        for inp in args.input:
            print(inp)
            dose = read_3ddose(inp)
            target = Target(np.array([0, 10, 0]), 1)
            import pprint
            pprint.pprint(dose_stats(dose, target))
    else:
        dose = read_3ddose(args.input[0])
        write_3ddose(args.output, dose)

    #    phant = read_egsphant(args.input)
    #    write_egsphant(args.output, phant)
    #print(paddick(dose, target))
    #print(dvh(dose, target))

