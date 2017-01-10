import sys
import math
import os

import numpy
from scipy.optimize import nnls
import tensorflow as tf

from py3ddose import read_3ddose, Dose, write_3ddose


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
        dose = read_3ddose(path)
        beams.append(dose)
    return beams


def scipy_nnls(beams, ideal):
    off_axes = [abs(i) * .2 for i in range(-374 // 2, 374 // 2)]
    d = 50
    return numpy.array([pow(oa / d, 2) for oa in off_axes])
    print(beams.shape, ideal.shape)
    return nnls(beams.T, ideal)[0]


def get_weights3(just_ys):
    sess = tf.InteractiveSession()
    size = just_ys[0].size
    X = tf.placeholder(tf.float32, shape=[size, len(just_ys)], name='X')
    W = tf.Variable(tf.truncated_normal(shape=[len(just_ys), 1], mean=.5, stddev=.25, dtype=tf.float32, seed=0, name='randomize_weights'))
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
        X: just_ys.T
    }
    steps = 10000
    for i in range(steps):
        print('Training step {} of {}'.format(i + 1, steps))
        s, w, _y, l, reg, m, var = sess.run([train_step, W, y, loss, regularization, mean, variance], feed_dict=data)
        print(w, reg, m, var)
    return sess.run(W)


def get_weights(beams, ideal):
    # r1 = 1000
    a = numpy.ones(374)
    for i in range(-186, 186):
        r = i * .2
        r1 = 10 + 40
        rp = math.sqrt(r * r + r1 * r1)
        a[i] = (r / rp) * (r / rp)
        print(i, a[i])
    return a
    zyslices = []
    ideal_distribution = ideal.doses[:, :, 50].flatten()
    zyslices = numpy.empty((len(beams), ideal_distribution.size))
    for i, beam in enumerate(beams):
        zyslices[i] = beam.doses[:, :, 50].flatten()
    # zyslices = numpy.ndarray(zyslices)
    print(zyslices.shape)

    sess = tf.InteractiveSession()
    X = tf.placeholder(tf.float32, shape=[ideal_distribution.size, len(beams)], name='X')
    yt = tf.placeholder(tf.float32, shape=[ideal_distribution.size], name='yt')
    W = tf.Variable(tf.truncated_normal(shape=[len(beams), 1], mean=.5, stddev=.25, dtype=tf.float32, seed=0, name='randomize_weights'))
    # b = tf.Variable(tf.truncated_normal(shape=[size], mean=.5, stddev=.25, dtype=tf.float32, seed=0, name='randomize_b'))
    sess.run(tf.global_variables_initializer())
    # weight the inputs
    y = tf.matmul(X, W) # + b
    # scale them
    max_dose = tf.reduce_max(y)
    y *= (1 / max_dose)
    regularization = -tf.reduce_sum(tf.minimum(W + 1, 0))
    # add error and regularization (but weight error less)
    loss = tf.reduce_sum(tf.abs(y - yt)) + regularization * 10000
    train_step = tf.train.GradientDescentOptimizer(.1).minimize(loss)
    data = {
        X: zyslices.T,
        yt: ideal_distribution
    }
    steps = 100000
    for i in range(steps):
        print('Training step {} of {}'.format(i + 1, steps))
        s, w, l, reg = sess.run([train_step, W, loss, regularization], feed_dict=data)
        print(w, l, reg)
    return sess.run(W)


def main2():
    # here we just try to make the skin dose the same, then we find the weights
    # here we try to make the sum of the skin dose 2 voxels deep
    beam_doses = get_beam_doses()
    just_ys = []
    total_beam = numpy.empty([len(beam_doses), beam_doses[0].doses.size])
    just_ys = numpy.empty([len(beam_doses), beam_doses[0].doses.shape[1]])
    for i, beam in enumerate(beam_doses):
        just_ys[i] = beam.doses[2:4, :, 49:52].sum(axis=0).sum(axis=1)
        total_beam[i] = beam.doses.flatten()
    ideal = numpy.full(just_ys.shape[1], numpy.amax(just_ys))
    weights = scipy_nnls(just_ys, ideal)
    print('Weights are')
    print(list(weights))
    final_beam = numpy.dot(total_beam.T, weights).reshape(beam_doses[0].doses.shape)
    write_3ddose('weighted.3ddose', Dose(beam_doses[0].boundaries, final_beam, beam_doses[0].errors))


def main3():
    beam_doses = get_beam_doses()
    just_ys = []
    total_beam = numpy.empty([len(beam_doses), beam_doses[0].doses.size])
    just_ys = numpy.empty([len(beam_doses), beam_doses[0].doses.shape[1]])
    for i, beam in enumerate(beam_doses):
        just_ys[i] = beam.doses[2, :, 50].sum(axis=0).sum(axis=1)
        total_beam[i] = beam.doses.flatten()
    weights = get_weights3(just_ys)
    final_beam = numpy.dot(total_beam.T, weights).reshape(beam_doses[0].doses.shape)
    write_3ddose('weighted.3ddose', Dose(beam_doses[0].boundaries, final_beam, beam_doses[0].errors))


def main():
    print('Getting ideal dose')
    ideal_dose = read_3ddose('ideal.3ddose')
    print('Getting beam doses')
    beam_doses = get_beam_doses()
    print('Calculating weights')
    weights = get_weights(beam_doses, ideal_dose)
    print('Weight shape is {}'.format(weights.shape))
    print('Building total beam')
    total_beam = []
    for beam in beam_doses:
        size = numpy.prod(beam.doses.shape)
        total_beam.append(beam.doses.reshape(size))
    total_beam = numpy.array(total_beam)
    print('Using weights on total beam of shape {}'.format(total_beam.shape))
    expected_distribution = numpy.dot(total_beam.T, weights).reshape(100, 100, 100)
    print('Writing weighted dose to file')
    write_3ddose('weighted.3ddose', Dose(ideal_dose.boundaries, expected_distribution, ideal_dose.errors))

if __name__ == '__main__':
    main2()
