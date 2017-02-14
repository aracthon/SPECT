import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
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

    # Convert labels {-1, 1}
    tmp  = labels

    labels[labels==0] = -1

    # Get the shape
    size, num_features = data.shape

    return (data, tmp, labels, size, num_features)

def main(argv=None):

    # Get the formulated data & information

    train_data_filename = '../data/SPECT.train.txt'
    test_data_filename = '../data/SPECT.test.txt'
    
    train_data, pr_train_labels, train_labels, train_size, num_features = formulate_data(train_data_filename)
    test_data, pr_test_labels, test_labels, test_size, num_features = formulate_data(test_data_filename)

    # Declare the input
    xData  = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
    yLabels = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    # Create Variables
    W = tf.Variable(tf.random_normal([num_features,1]))
    b = tf.Variable(tf.random_normal([1]))

    # Declare the output
    yRaw = tf.add(tf.matmul(xData, W), b)

    # Model the loss function
    reg_para = tf.constant([0.01])
    svmC = 1
    l2_norm = tf.reduce_sum(tf.square(W))
    classification_term = svmC * tf.reduce_sum(tf.maximum(tf.zeros([40,1]), 
                                                1 - yLabels*yRaw));
    regularization_loss = tf.mul(reg_para, l2_norm)
    loss = tf.add(classification_term, tf.mul(regularization_loss, reg_para))

    # Model the prediction function
    prediction = tf.sign(yRaw)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, yLabels), dtype=tf.float32))

    # Model the optimizer
    # learning_rate = tf.constant([0.001])
    learning_rate = 0.001
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Define training parameters
    batch_size = 40
    train_epochs = 1000

    # Count true positives, true negatives, false positives and false negatives.
    tp = tf.count_nonzero(prediction * train_labels)
    tn = tf.count_nonzero((prediction - 1) * (train_labels - 1))
    fp = tf.count_nonzero(prediction * (train_labels - 1))
    fn = tf.count_nonzero((prediction - 1) * train_labels)
        
    # Calculate accuracy, precision, recall and F1 score.
    # accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    with tf.Session() as sess:

        # Initialize all variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # Define metric parameters
        loss_vec = []
        train_accuracy = []
        test_accuracy = []
        train_precision = []
        train_recall = []
        test_precision = []
        test_recall = []

        # Training
        for epoch in xrange(train_epochs):

            # Randomize batch
            rand_index = np.random.choice(len(train_data), size=batch_size)
            rand_x = train_data[rand_index]
            rand_y = train_labels[rand_index]
            sess.run(optimizer, feed_dict={xData: rand_x, yLabels: rand_y})

            # Append metric data
            tmp_loss = sess.run(loss, feed_dict={xData: rand_x, yLabels: rand_y})
            loss_vec.append(tmp_loss)

            train_acc_tmp = sess.run(accuracy, feed_dict={xData: train_data, yLabels: train_labels})
            train_accuracy.append(train_acc_tmp)

            test_acc_tmp = sess.run(accuracy, feed_dict={xData: test_data, yLabels: test_labels})
            test_accuracy.append(test_acc_tmp)

            # Append precision & recall

            # train_acc_prediction = sess.run(train_precision, feed_dict={xData: train_data, yLabels: train_labels})
            # train_precision.append(train_acc_prediction)
            # train_acc_recall = sess.run(train_recall, feed_dict={xData: train_data, yLabels: train_labels})
            # train_recall.append(train_acc_recall)

            # test_acc_prediction = sess.run(test_precision, feed_dict={xData: test_data, yLabels: test_labels})
            # test_precision.append(test_acc_prediction)
            # test_acc_recall = sess.run(test_recall, feed_dict={xData: test_data, yLabels: test_labels})
            # test_recall.append(test_acc_recall)

            # indices = sess.run(indices, feed_dict={xData: test_data, yLabels: test_labels})

            # s = 0
            # for index in indices:
            #     if test_labels[index] == 1:
            #         s += 1
            # precision = s * 1.0 / len(indices)
            # test_precision.append(precision)

            train_prediction = sess.run(prediction, feed_dict={xData: train_data, yLabels: train_labels})
            train_prediction[train_prediction==-1] = 0
            train_precision = sess.run(precision, feed_dict={prediction: train_prediction, train_labels:train_labels})


        # Plot train/test accuracies
        plt.plot(train_accuracy, 'k-', label='Training Accuracy')
        plt.plot(test_accuracy, 'r--', label='Test Accuracy')
        plt.title('Train & Test Set Accuracies')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()

        # Plot precision % recall
        plt.plot(train_precision, 'k-', label='Training Precision')
        # plt.plot(train_recall, 'r--', label='Training Recall')
        plt.title('Train Precision & Recall')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()

        # plt.plot(test_precision, 'k-', label='Test Precision')
        # plt.plot(test_recall, 'r--', label='Test Recall')
        # plt.title('Train Precision & Recall')
        # plt.xlabel('Generation')
        # plt.ylabel('Accuracy')
        # plt.legend(loc='lower right')
        # plt.show()

        # Plot loss over time
        plt.plot(loss_vec, 'k-')
        plt.title('Loss per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.show()

if __name__ == '__main__':
    main()