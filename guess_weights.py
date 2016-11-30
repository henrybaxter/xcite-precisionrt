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

d = 'henry-2-1e10'
beams = []
z = 2
x = 50
paths = []
for i in range(374):
    path = os.path.join(d, 'dose{}.3ddose'.format(i))
    paths.append(path)

for path in paths:
    print('Reading {}'.format(path))
    dose = read_3ddose(path)
    just_y = dose.doses[z, :, x]
    beams.append(just_y)

beams = numpy.array(beams)
print('Constructed all beam contributions')
target_dose = numpy.amax(beams)
print('Target skin dose is {}'.format(target_dose))
target_distribution = numpy.full(100, target_dose)
print('Target distribution constructed')

# result = numpy.array([0, 0, 0, 0, 100])
# print(beams.T.shape, target_distribution.shape)
print('Performing Non-negative least squares fit')
result = nnls(beams.T, target_distribution)
weights = result[0]
print('Weights retrieved:\n{}'.format(weights))

# ok now we have to take all the originals and weight them!
#result = beams.T * weights
#print('Max is {}'.format(numpy.amax(doses)))

print('Writing dose')
dose = Dose(dose.boundaries, result, dose.errors)
weight_3ddose(paths, 'weighted.3ddose', weights)
