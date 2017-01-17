import os
import platform
import hashlib
import json

import matplotlib
if platform.system() == 'Darwin':
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.spatial.distance import pdist

from . import py3ddose


def get_beam_doses():
    beams = []
    for path in dose_paths():
        dose = py3ddose.read_3ddose(path)
        beams.append(dose)
    return beams


def get_combined():
    doses = []
    for dose in get_beam_doses():
        doses.append(dose.doses)
    return np.array(doses).sum(axis=0)


def maximally_distant(points, n):
    distances = {}
    import itertools
    import random
    possibles = list(itertools.combinations(points, n))
    for ps in random.sample(possibles, min(len(possibles), 200)):
        d = []
        for p1, p2 in zip(ps, ps[1:] + ps[:1]):
            d.append(np.sqrt(np.sum(np.square(p1 - p2))))
        distances[min(d)] = ps
    return distances[max(distances.keys())]


def get_manual(possibles):
    # just pick one
    result = [possibles[0][0]]
    for possible in possibles[1:]:
        for i in range(3):
            diff = np.sum(np.square(possible - result[-1]), axis=1)
            indices = np.where(diff == diff.max())
            result.append(possible.pop(indices[0][0]))
    return result


def get_manual2(possibles):
    # how do we get the length of the path?
    # distance between each
    result = []
    specials = []
    for i, possible in enumerate(possibles):
        points = np.concatenate([possible, possible[:1]])
        length = np.sqrt(np.sum(np.square(np.diff(points, axis=0))))
        n_points = int(length) // 5
        if n_points == 0:
            specials.append(possible)
            continue
        np.roll(points, i)
        # now we want to get them equidistant
        # so we choose the segment lenght
        seglength = length / n_points
        last = 0
        for j in range(len(points)):
            l = np.sqrt(np.sum(np.square(np.diff(points[last:j], axis=0))))
            if l >= seglength:
                last = j
                result.append(points[j])
    import itertools
    distances = {}
    prod = list(itertools.product(*specials))
    import random
    for ps in random.sample(prod, 1000):
        d = np.sum(pdist(ps))
        distances[d] = ps
    result.extend(distances[min(distances.keys())])
    return result

    return []
    import itertools
    import random
    possibles = list(itertools.chain(*possibles))
    distances = {}
    for i in range(100):
        ps = random.sample(possibles, 30)
        d = []
        for p1, p2 in zip(ps, ps[1:] + ps[:1]):
            d.append(np.sqrt(np.sum(np.square(p1 - p2))))
        distances[sum(d)] = ps
    return distances[max(distances.keys())]


DEFAULT_LEVELS = [5.0, 10.0, 20.0, 30.0, 50.0, 70.0, 80.0, 90.0]


async def plot(egsphant_path, dose_path, target, output_slug, levels=DEFAULT_LEVELS):
    print('Plotting at dose path', dose_path, output_slug)
    iso = target.isocenter.tolist()
    rad = target.radius
    inputs = [egsphant_path, dose_path, iso, rad, output_slug, levels]
    base = hashlib.md5(json.dumps(inputs).encode('utf-8')).hexdigest()
    # this actually generates three files. let's make it functional? or what...
    dose = py3ddose.read_3ddose(dose_path)
    print('max dose is', np.max(dose.doses))
    phantom = py3ddose.read_egsphant(egsphant_path)
    centers = [(b[1:] + b[:-1]) / 2 for b in dose.boundaries]
    isocenter = np.argmin(np.abs(centers - target.isocenter[:, np.newaxis]), axis=1)
    # reference_dose = dose.doses[tuple(isocenter)]
    # highest = np.unravel_index(dose.doses.argmax(), dose.doses.shape)
    reference_dose = np.max(dose.doses)
    normalized = dose.doses / reference_dose * 100

    Z_AXIS, Y_AXIS, X_AXIS = range(3)
    axis_names = {
        Z_AXIS: 'z',
        Y_AXIS: 'y',
        X_AXIS: 'x'
    }
    plots = []
    for i, z_axis in enumerate([X_AXIS, Y_AXIS, Z_AXIS]):
        if z_axis == X_AXIS:
            x_axis = Y_AXIS
            y_axis = Z_AXIS
            invert_y = True
            D = normalized[:, :, isocenter[z_axis]:isocenter[z_axis] + 1]
            densities = phantom.densities[:, :, isocenter[z_axis]]
        elif z_axis == Y_AXIS:
            x_axis = Z_AXIS
            y_axis = X_AXIS
            invert_y = True
            D = normalized[:, isocenter[z_axis]:isocenter[z_axis] + 1, :].T
            densities = phantom.densities[:, isocenter[z_axis], :]
        elif z_axis == Z_AXIS:
            x_axis = X_AXIS
            y_axis = Y_AXIS
            invert_y = True
            D = normalized[isocenter[z_axis]:isocenter[z_axis] + 1, :, :]
            densities = phantom.densities[isocenter[z_axis], :, :]

        x_name = axis_names[x_axis]
        y_name = axis_names[y_axis]

        slug = 'contour_{}_{}'.format(x_name, y_name)


        # bottom axis is Y
        X = centers[x_axis]
        Y = centers[y_axis]

        D = np.mean(D, axis=z_axis)

        plt.figure()
        extents = [
            np.min(dose.boundaries[x_axis]),
            np.max(dose.boundaries[x_axis]),
            np.min(dose.boundaries[y_axis]),
            np.max(dose.boundaries[y_axis])
        ]

        plt.imshow(densities,
                   extent=extents, cmap='gray', vmin=0.2, vmax=1.5,
                   interpolation='nearest')
        if invert_y:
            plt.gca().invert_yaxis()
        cs = plt.contour(X, Y, D, levels=levels, cmap=cm.jet, linewidths=1)
        paths = []
        for i, cc in enumerate(cs.collections):
            for j, pp in enumerate(cc.get_paths()):
                points = []
                for k, vv in enumerate(pp.iter_segments()):
                    points.append(vv[0])
                paths.append(points)
        plt.clabel(cs, fontsize=8, fmt='%2.0f')

        filename = slug + '.pdf'
        subfolder = os.path.join('contours', output_slug)
        os.makedirs(subfolder, exist_ok=True)
        path = os.path.join(subfolder, filename)
        plt.savefig(path)
        plane = x_name + y_name
        plots.append({
            'output_slug': output_slug,
            'plane': plane,
            'slug': slug,
            'path': path,
            'name': '{} {}'.format(output_slug.replace('_', ' ').title(), plane.upper())
        })
    return plots


if __name__ == '__main__':
    target = py3ddose.Target(np.array([-10, 20, 0], 1))
    egsphant_path = 'cylindricalp.egsphant'
    dose_path = 'dose.3ddose'
    output_dir = 'test_contours'
    os.makedirs(output_dir, exist_ok=True)
    plot(egsphant_path, dose_path, target, output_dir)
