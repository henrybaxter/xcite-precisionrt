import os
import numpy

from py3ddose import read_3ddose


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
contributions = []
for i in range(5):
    try:
        dose = read_3ddose(os.path.join(d, 'dose{}.3ddose'.format(i)))
    except:
        print('Skipping not there...')
        continue
    contributions.append(dose.doses.sum(axis=(0, 2)))
    print('Got {}\'s contribution'.format(i))

contributions = numpy.array(contributions)
print(len(contributions[0]))
result = numpy.linalg.lstsq(contributions, numpy.ones(len(contributions[0])))
print(result)
#print(result)
