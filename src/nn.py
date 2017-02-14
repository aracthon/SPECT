# Implementing a one-layer Neural Network
#---------------------------------------
#
# We will illustrate how to create a one hidden layer NN
#
# We will use the iris data for this exercise
#
# We will build a one-hidden layer neural network
#  to predict the fourth attribute, Petal Width from
#  the other three (Sepal length, Sepal width, Petal length).

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.metrics import classification_report
 
ops.reset_default_graph()

def extract_data(filename):

    out = np.loadtxt(filename, delimiter=',');

    # Arrays to hold the labels and feature vectors.
    labels = out[:,0]
    labels = labels.reshape(labels.size,1)
    fvecs = out[:,1:]

    # Return a pair of the feature matrix and the one-hot label matrix.
    return fvecs,labels

def formulate_data(filename):

    # Extract data
    data, labels = extract_data(filename)

    # Get the shape
    size, num_features = data.shape

    return (data, labels, size, num_features)

train_data_filename = '../data/SPECT.train.txt'
test_data_filename = '../data/SPECT.test.txt'

train_data, train_labels, train_size, num_features = formulate_data(train_data_filename)
test_data, test_labels, test_size, num_features = formulate_data(test_data_filename)

# Create graph session 
sess = tf.Session()

# Declare batch size
batch_size = 50

# Initialize placeholders
xData = tf.placeholder(shape=[None, 22], dtype=tf.float32)
yLabels = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Number of hidden nodes
hidden_layer1_nodes = 20
hidden_layer2_nodes = 10

# Hidden layer1
W1 = tf.Variable(tf.random_normal(shape=[22,hidden_layer1_nodes]))
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer1_nodes]))

# Hidden layer2
W2 = tf.Variable(tf.random_normal(shape=[hidden_layer1_nodes,hidden_layer2_nodes]))
b2 = tf.Variable(tf.random_normal(shape=[hidden_layer2_nodes]))

# Output layer
W3 = tf.Variable(tf.random_normal(shape=[hidden_layer2_nodes,1]))
b3 = tf.Variable(tf.random_normal(shape=[1]))

# Build model
hidden_layer1_output = tf.nn.relu(tf.add(tf.matmul(xData, W1), b1))
hidden_layer2_output = tf.nn.relu(tf.add(tf.matmul(hidden_layer1_output, W2), b2))
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_layer2_output, W3), b3))

# Calculate Minimum Square Error
loss = tf.reduce_mean(tf.square(yLabels - final_output))

# Declare optimizer
learning_rate = 0.001
training_epochs = 1000
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
test_loss = []

for epoch in xrange(training_epochs):

    # Randomize
    rand_index = np.random.choice(len(train_data), size=batch_size)
    rand_x = train_data[rand_index]
    rand_y = train_labels[rand_index]

    # Training
    sess.run(optimizer, feed_dict={xData: rand_x, yLabels: rand_y})

    temp_loss = sess.run(loss, feed_dict={xData: rand_x, yLabels: rand_y})
    loss_vec.append(np.sqrt(temp_loss))
    
    # Testing
    test_temp_loss = sess.run(loss, feed_dict={xData: test_data, yLabels: test_labels})
    test_loss.append(np.sqrt(test_temp_loss))

predicted = sess.run(final_output, feed_dict={xData: test_data})
predicted[predicted!=0] = 1
# print predicted.T
print classification_report(test_labels, predicted)

# Plot loss (MSE) over time
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()