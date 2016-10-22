from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
import numpy as np

class SvmModel:
    def __init__(self):
        self.classifier = svm.SVC()
        self.fitted = False

    def fit(self, training_data):
        '''
        Fit the model to a set of data.
        training_data: [([feature_vec], label)]
        '''
        features = [ entry[0] for entry in training_data ]
        labels = [ entry[1] for entry in training_data ]
        self.classifier.fit(features, labels)
        self.fitted = True

    def predict(self, data):
        '''
        Classify a set of data using the model.
        data: [[feature_vec]]
        '''
        return self.classifier.predict(data)

    def crossvalidate(self, training_data, percent):
        '''
        Cross-validate the data.
        training_data: [([feature_vec], label)]
        ratio: ratio of training 
        '''
        # split the data
        split_idx = int(percent * len(training_data))
        trainingset = training_data[:split_idx]
        validateset = training_data[split_idx:]

        print("Training set size: {0}".format(len(trainingset)))
        print("Validation set size: {0}".format(len(validateset)))

        # train with the training part
        print("Fitting...")
        self.fit(trainingset)
        # make predictions on the other part
        print("Predicting...")
        predictions = self.predict([entry[0] for entry in validateset])
        # calculate the accuracy
        return accuracy_score(predictions, [entry[1] for entry in validateset])

    def probability(self):
        dist = self.classifier.decision_function()
        max_pos = max(dist)
        max_neg = min(dist)
        prob = map(lambda x: x/max_neg if x < 0 else x/max_pos, dist)

class GaussianNBModel:
    def __init__(self):
        self.classifier = GaussianNB()
        self.fitted = False

    def fit(self, training_data):
        '''
        Fit the model to a set of data.
        training_data: [([feature_vec], label)]
        '''
        features = np.array([ entry[0] for entry in training_data ])
        labels = np.array([ entry[1] for entry in training_data ])
        self.classifier.fit(features, labels)
        self.fitted = True

    def predict(self, data):
        '''
        Classify a set of data using the model.
        data: [[feature_vec]]
        '''
        return self.classifier.predict(data)

    def crossvalidate(self, training_data, percent):
        '''
        Cross-validate the data.
        training_data: [([feature_vec], label)]
        ratio: ratio of training 
        '''
        # split the data
        split_idx = int(percent * len(training_data))
        trainingset = training_data[:split_idx]
        validateset = training_data[split_idx:]

        print("Training set size: {0}".format(len(trainingset)))
        print("Validation set size: {0}".format(len(validateset)))

        # train with the training part
        print("Fitting...")
        self.fit(trainingset)
        # make predictions on the other part
        print("Predicting...")
        predictions = self.predict([entry[0] for entry in validateset])
        # calculate the accuracy
        return accuracy_score(predictions, [entry[1] for entry in validateset])

    def probability(self):
        dist = self.classifier.decision_function()
        max_pos = max(dist)
        max_neg = min(dist)
        prob = map(lambda x: x/max_neg if x < 0 else x/max_pos, dist)
