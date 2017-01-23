import os
import hashlib
import json
import asyncio

from . import matplotlibstub  # NOQA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.spatial.distance import pdist

from . import py3ddose
from . import egsphant


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
    with open(egsphant_path) as fp:
        phantom = egsphant.read_egsphant(fp)
    centers = [(b[1:] + b[:-1]) / 2 for b in dose.boundaries]
    translated = [c - target.isocenter[i] for i, c in enumerate(centers)]
    xx, yy, zz = np.meshgrid(*translated, indexing='ij')
    d2 = np.square(xx) + np.square(yy) + np.square(zz)
    print(d2.shape)
    isocenter = np.unravel_index(np.argmin(d2), d2.shape)
    print('isocenter is', isocenter)
    # isocenter = np.argmin(np.abs(centers - target.isocenter[:, np.newaxis]), axis=1)
    # reference_dose = dose.doses[tuple(isocenter)]
    # highest = np.unravel_index(dose.doses.argmax(), dose.doses.shape)
    reference_dose = np.max(dose.doses)
    normalized = dose.doses / reference_dose * 100

    X_AXIS, Y_AXIS, Z_AXIS = range(3)
    axis_names = {
        X_AXIS: 'x',
        Y_AXIS: 'y',
        Z_AXIS: 'z'
    }
    plots = []
    for i, z_axis in enumerate([X_AXIS, Y_AXIS, Z_AXIS]):
        print()
        print()
        if z_axis == X_AXIS:
            x_axis = Y_AXIS
            y_axis = Z_AXIS
            z = isocenter[z_axis]
            D = normalized[z - 1:z + 2, :, :]
            densities = phantom.densities[z, :, :]
        elif z_axis == Y_AXIS:
            x_axis = X_AXIS
            y_axis = Z_AXIS
            z = isocenter[z_axis]
            D = normalized[:, z - 1:z + 2, :]
            densities = phantom.densities[:, z, :]
        elif z_axis == Z_AXIS:
            x_axis = X_AXIS
            y_axis = Y_AXIS
            z = isocenter[z_axis]
            D = normalized[:, :, z - 1:z + 2]
            densities = phantom.densities[:, :, z]
        x_name = axis_names[x_axis]
        y_name = axis_names[y_axis]
        print(x_name, y_name, z)
        print(D.shape, densities.shape)
        print('centers', [c.shape for c in centers])
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
        print(densities.shape)
        print(densities)
        print('X.shape', X.shape)
        print('Y.shape', Y.shape)
        plt.imshow(densities,
                   extent=extents, cmap='gray', vmin=0.2, vmax=1.5,
                   interpolation='nearest')
        # if invert_y:
        plt.gca().invert_yaxis()
        cs = plt.contour(X, Y, D.T, levels=levels, cmap=cm.jet, linewidths=1)
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
        plt.xlabel(x_name + ' (cm)')
        plt.ylabel(y_name + ' (cm)')
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


async def main():
    target = py3ddose.Target(np.array([0, 0, 10]), 1)
    egsphant_path = 'phantoms/20cm-long-40cm-wide-2mm-cylinder.egsphant'
    dose_path = 'dose/stationary.3ddose'
    output_dir = 'test_contours'
    os.makedirs(output_dir, exist_ok=True)
    return await plot(egsphant_path, dose_path, target, output_dir)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
