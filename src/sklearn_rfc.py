import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve

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

	rfc = RandomForestClassifier(n_estimators=100)
	lr = LogisticRegression()
	gnb = GaussianNB()
	svc = LinearSVC(C=1.0)
	
	plt.figure(figsize=(10, 10))
	ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

	ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
	
    rfc.fit(train_data, train_labels)
    if hasattr(clf, "predict_proba"):
        prob_pos = rfc.predict_proba(test_data)[:, 1]
    else:  # use decision function
        prob_pos = rfc.decision_function(test_data)
        prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(test_labels, prob_pos, n_bins=50)

    # ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
    #          label="%s" % (name, ))
	    print prob_pos
	ax1.set_ylabel("Fraction of positives")
	ax1.set_ylim([-0.05, 1.05])
	ax1.legend(loc="lower right")
	ax1.set_title('Calibration plots  (reliability curve)')

	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()






