import logging

import numpy as np

logger = logging.getLogger(__name__)


def skin_indices(length=20, radius=20, voxel=0.2, sweep=10):
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
    print('Total voxels: {}'.format(n_x * n_y * n_z))
    x_boundaries = np.linspace(x_min, x_max, n_x + 1)
    y_boundaries = np.linspace(y_min, y_max, n_y + 1)
    z_boundaries = np.linspace(z_min, z_max, n_z + 1)
    x_centers = (x_boundaries[1:] + x_boundaries[:-1]) / 2
    y_centers = (y_boundaries[1:] + y_boundaries[:-1]) / 2
    z_centers = (z_boundaries[1:] + z_boundaries[:-1]) / 2
    xx, yy, zz = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    # we remove one voxel on either side as the 'air'
    r2_outer = np.square(radius - voxel)
    r2_inner = np.square(radius - 3 * voxel)
    r2 = np.square(yy) + np.square(zz - radius)
    ssd = 50
    depth = 10
    c_width = 70
    c_radius = c_width / 2
    skin_x_max = 7
    skin_x_min = -7
    angle = sweep / 2 * (np.pi / 180)
    skin = np.all([
        r2 <= r2_outer,
        r2 > r2_inner,
        xx <= skin_x_max,
        xx >= skin_x_min,
        zz <= (1 if sweep == 10 else 5)
    ], axis=0)
    return skin


def target_indices(dose, target):
    centers = [(b[1:] + b[:-1]) / 2 for b in dose.boundaries]
    translated = [c - target.isocenter[i] for i, c in enumerate(centers)]
    xx, yy, zz = np.meshgrid(*translated, indexing='ij')
    d2 = np.square(xx) + np.square(yy) + np.square(zz)
    r2 = np.square(target.radius)
    return d2 < r2


def target_to_skin_ratio(dose, target):
    """
    So we look at indices in the skin, we find the doses for each, and then we work with them
    Don't worry about tensor flow, just calculate the skin target ratio

    """
    skin_mean = np.mean(dose.doses[skin_indices()])
    logger.info('Skin mean is {}'.format(skin_mean))
    target_mean = np.mean(dose.doses[target_indices(dose, target)])
    logger.info('Target mean is {}'.format(target_mean))
    result = target_mean / skin_mean
    logger.info('Skin target ratio is {}'.format(result))
    return result


def stationary_weights_nnls(doses, target):
    skin_doses = []
    for dose in doses:
        skin_doses.append(dose.doses[skin_indices()])
    skin_doses = np.array(skin_doses)
    from scipy.optimize import nnls
    ideal = np.full(skin_doses[0].size, np.amax(skin_doses))
    weights = nnls(skin_doses, ideal)
    logger.info('Weights are {}'.format(weights))
    return weights


def stationary_weights_tf(doses, target):
    skin_doses = []
    for dose in doses:
        skin_doses.append(dose.doses[skin_indices()])
    skin_doses = np.array(skin_doses)
    import tensorflow as tf
    sess = tf.InteractiveSession()
    X = tf.placeholder(tf.float32, shape=[skin_doses.shape[1], skin_doses.shape[0]], name='X')
    W = tf.Variable(tf.truncated_normal(shape=[skin_doses[0], 1], mean=2, stddev=.25, dtype=tf.float32, seed=0, name='randomize_weights'))
    sess.run(tf.global_variables_initializer())
    y = tf.matmul(X, W)
    y /= tf.reduce_max(y)
    # let's try and make each of them the *same*
    # and let's also try to keep the weights above 1 and below a bazillion
    regularization = -tf.reduce_sum(tf.minimum(W - 1, 0))
    mean, variance = tf.nn.moments(y, axes=[0])
    loss = variance * 100 + regularization
    train_step = tf.train.GradientDescentOptimizer(.001).minimize(loss)
    data = {
        X: skin_doses.T
    }
    steps = 10000
    for i in range(steps):
        print('Training step {} of {}'.format(i + 1, steps))
        s, w, _y, l, reg, m, var = sess.run([train_step, W, y, loss, regularization, mean, variance], feed_dict=data)
        print(w, reg, m, var)
    return sess.run(W)


if __name__ == '__main__':
    print('hello')
