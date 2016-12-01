import os
import numpy
from scipy.optimize import nnls

from py3ddose import read_3ddose, Dose, write_3ddose, weight_3ddose


"""
import tensorflow as tf
sess = tf.InteractiveSession()
dose = read_3ddose('example.3ddose')
n_beamlets = 10
shape = [n_beamlets] + list(dose.doses.shape)
print(shape)
x = tf.placeholder(tf.float32, shape=shape)
yt = tf.placeholder(tf.float32, shape=dose.doses.shape)
W = tf.Variable(tf.zeros([n_beamlets]))
b = tf.Variable(tf.zeros([n_beamlets]))
sess.run(tf.global_variables_initializer())
y = tf.matmul(x, W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, yt))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
data = {
    'x': [dose.doses for i in range(n_beamlets)],
    'y': dose.doses
}
for i in range(1000):
    train_step.run(feed_dict=data)

correct_prediction = tf.equal(tf.argmax())
"""

# get a bunch of 3ddose files
# sum them together, and we have a problem

ideal_path = 'ideal.3ddose'
d = 'henry-2-1e10'
beams = []
paths = []
for i in range(374):
    path = os.path.join(d, 'dose{}.3ddose'.format(i))
    paths.append(path)

ideal = read_3ddose(ideal_path)
ideal_distribution = ideal.doses
size = numpy.prod(ideal_distribution.shape)
ideal_distribution = ideal_distribution.reshape(size)

for path in paths:
    print('Reading {}'.format(path))
    beams.append(read_3ddose(path).doses.reshape(size))

beams = numpy.array(beams)

max_dose = numpy.amax(beams)
print('Max dose found: {:.g}'.format(max_dose))

# now we need to normalize as a percentage of max dose
beams = numpy.multiply(beams, 1 / max_dose)

# now we want the max dose in the target, and the min dose around.
# and we've got that now.

print('Performing Non-negative least squares fit')
result = nnls(beams.T, ideal_distribution)
weights = result[0]
print('Weights retrieved:\n{}'.format(weights))

weighted_beams = (beams.T * weights).T
expected_distribution = weighted_beams.sum(axis=0)

# write to weighted
# TODO correct errors
write_3ddose('weighted.3ddose', Dose(ideal.boundaries, expected_distribution, ideal.errors))


# ok now we have to take all the originals and weight them!
# result = beams.T * weights
# print('Max is {}'.format(numpy.amax(doses)))

print('Writing dose')
weight_3ddose(paths, 'weighted.3ddose', weights)
