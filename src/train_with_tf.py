import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import os

src_path = os.path.abspath(os.path.dirname(sys.argv[0]))
data_path = '../data/'

training_data = np.loadtxt(data_path + 'SPECT.train.txt', delimiter=',');
test_data = np.loadtxt(data_path + 'SPECT.test.txt', delimiter=',');

# Format the data
# X: 80x22, Y:187x22

training_data_y = training_data[:,0]
training_data_x = training_data[:,1:23]

test_data_y = test_data[:,0]
test_data_x = test_data[:,1:23]

# Build the model
x = tf.placeholder(tf.float32, [None, 22])
W = tf.Variable(tf.zeros([22, 1]))
b = tf.Variable(tf.zeros([1,1]))

t = tf.matmul(x, W) + b

y = tf.nn.softmax(t)

# Calculate the cross entropy
y_ = tf.placeholder(tf.float32, [None, None])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Define Gradient Descent Optimizer
learning_rate = 0.5
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# Initialization
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
rng = np.random.RandomState(100)

# Define variables
batch_size = 100
epochs = 1000

# Training
for epoch in xrange(epochs):
	sess.run(train_step, feed_dict = {x: training_data_x, y: training_data_y})

# Summary
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Test Data
print(sess.run(accuracy, feed_dict={x: test_data_x}))






