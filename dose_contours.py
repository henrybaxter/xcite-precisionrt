import os

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np

import py3ddose


def dose_paths():
    paths = []
    d = 'henry-2-1e10'
    for i in range(374):
        path = os.path.join(d, 'dose{}.3ddose'.format(i))
        paths.append(path)
    return paths


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
        d = 0
        for p1, p2 in zip(ps, ps[1:] + ps[:1]):
            d += np.sqrt(np.sum(np.square(p1 - p2)))
        distances[d] = ps
    return distances[max(distances.keys())]


if __name__ == '__main__':
    target = py3ddose.Target(np.array([-10, 20, 0]), 1)
    dose = py3ddose.read_3ddose('combined.3ddose')
    phantom = py3ddose.read_egsphant('cylindricalp.egsphant')
    centers = [(b[1:] + b[:-1]) / 2 for b in dose.boundaries]
    isocenter = np.argmin(np.abs(centers - target.isocenter[:, np.newaxis]), axis=1)
    print(isocenter)
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
        print(x_axis, y_axis, z_axis)
        print('Generating Figure {} ({}-{} plane)'.format(i + 1, x_name, y_name))

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
        cs = plt.contour(X, Y, D, levels=[10.0, 20.0, 30.0, 50.0, 70.0, 80.0, 90.0], linewidths=2)
        # plt.clabel(cs, fmt='%2.0f')
        # pick three spots for each label at random? no
        paths = []
        for i, cc in enumerate(cs.collections):
            for j, pp in enumerate(cc.get_paths()):
                points = []
                for k, vv in enumerate(pp.iter_segments()):
                    points.append(vv[0])
                paths.append(maximally_distant(points, 3))
        # choose
        manual = []
        for path in paths:
            manual.extend(path)
        plt.clabel(cs, manual=manual, fmt='%2.0f')

        """
        plt.figure()
        cs = plt.contour(normalized[:, :, 50], origin='upper') #, levels=[10.0, 20.0, 30.0, 50.0, 70.0, 80.0, 90.0])
        plt.clabel(cs, fmt='%2.0f')
        """
        plt.show()
        #plt.savefig('contour_{}_{}.pdf'.format(x_name, y_name))
        
