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

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[22, 44, 22],
                                            model_dir="/tmp/a_spect_model")

# Fit model.
classifier.fit(x=training_data_x,
               y=training_data_y,
               steps=2000)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_data_x,
                                     y=test_data_y)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))



