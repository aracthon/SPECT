import tensorflow as tf
import numpy as np
import scipy.io as io
from matplotlib import pyplot as plt

# Global variables.
BATCH_SIZE = 16  # The number of training examples to use per training step.

def extract_data(filename, delimiter=','):

    out = np.loadtxt(filename, delimiter=delimiter);

    # Arrays to hold the labels and feature vectors.
    labels = out[:,0]
    labels = labels.reshape(labels.size,1)
    fvecs = out[:,1:]

    # Return a pair of the feature matrix and the one-hot label matrix.
    return fvecs,labels


def main(argv=None):
    # Be verbose?
    verbose = True

    # Plot? 
    plot = False
    
    # Get the data.
    train_data_filename = '../data/SPECT.train.txt'
    test_data_filename = '../data/SPECT.test.txt'

    # Extract it into numpy matrices.
    train_data, train_labels = extract_data(train_data_filename)
    test_data, test_labels = extract_data(test_data_filename)

    # Convert labels to +1,-1
    train_labels[train_labels==0] = -1
    test_labels[test_labels==0] = -1

    # Get the shape of the training data.
    train_size, num_features = train_data.shape
    test_size, num_features = test_data.shape

    # Get the number of epochs for training.
    num_epochs = 10

    # Get the C param of SVM
    svmC = 0.1#FLAGS.svmC

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    x = tf.placeholder("float", shape=[None, num_features])
    y = tf.placeholder("float", shape=[None,1])

    # Define and initialize the network.

    # These are the weights that inform how much each feature contributes to
    # the classification.
    W = tf.Variable(tf.zeros([num_features,1]))
    b = tf.Variable(tf.zeros([1]))
    y_raw = tf.matmul(x,W) + b

    # Optimization.
    regularization_loss = 0.5*tf.reduce_sum(tf.square(W)) 
    hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([BATCH_SIZE,1]), 
        1 - y*y_raw));
    svm_loss = regularization_loss + svmC*hinge_loss;
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(svm_loss)

    # Evaluation.
    predicted_class = tf.sign(y_raw);
    correct_prediction = tf.equal(y,predicted_class)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Create a local session to run this computation.
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
        if verbose:
            print 'Initialized!'
            print
            print 'Training.'

        # Iterate and train.
        for step in xrange(num_epochs * train_size // BATCH_SIZE):
            if verbose:
                print step,

            offset = (step * BATCH_SIZE) % train_size
            batch_data = train_data[offset:(offset + BATCH_SIZE), :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            train_step.run(feed_dict={x: batch_data, y: batch_labels})
            print 'loss: ', svm_loss.eval(feed_dict={x: batch_data, y: batch_labels})

            if verbose and offset >= train_size-BATCH_SIZE:
                print

        # Give very detailed output.
        if verbose:
            print
            print 'Weight matrix.'
            print s.run(W)
            print
            print 'Bias vector.'
            print s.run(b)
            print
            print "Applying model to first test instance."
            print
            
        print "Accuracy on train:", accuracy.eval(feed_dict={x: test_data, y: test_labels})
        
        # test
    
if __name__ == '__main__':
    main()
