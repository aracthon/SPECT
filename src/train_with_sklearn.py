import numpy as np
import sklearn
import sklearn.linear_model
import sys
import os

src_path = os.path.abspath(os.path.dirname(sys.argv[0]))
data_path = '../data/'

training_data = np.loadtxt(data_path + 'SPECT.train.txt', delimiter=',');
test_data = np.loadtxt(data_path + 'SPECT.test.txt', delimiter=',');

# print(test_data)

training_data_y = training_data[:,0]
training_data_x = training_data[:,1:23]

test_data_y = test_data[:,0]
test_data_x = test_data[:,1:23]

# print training_data_y.shape
# print training_data_x.shape

from sklearn.svm import SVC
from sklearn.metrics import classification_report

clf = SVC()
clf.fit(training_data_x, training_data_y) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=True)

real_result = test_data_y
prediction = clf.predict(test_data_x)

report = classification_report(test_data_y, prediction)
print(report)
print clf.score(test_data_x, test_data_y)