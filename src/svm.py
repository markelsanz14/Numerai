from sklearn import svm
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

class SvmModel:

    def __init__(self, training_set):
        self.training_set = training_set
        self.classifier = svm.SVC()

    def fit(self):
        features = [entry[0] for entry in self.training_set]
        labels = [entry[1] for entry in self.training_set]
        self.classifier.fit(features, labels)

    def predict(self, test_set):
        features = [entry[0] for entry in test_set]
        labels = [entry[1] for entry in test_set]
        predictions = self.classifier.predict(features)
        return accuracy_score(labels, predictions)

    def probability(self):
        dist = self.classifier.decision_function()
        max_pos = max(dist)
        max_neg = min(dist)
        prob = map(lambda x: x/max_neg if x < 0 else x/max_pos, dist)

        #y_true = [0, 0, 1, 1]
        #y_pred = [[.9, .1], [.8, .2], [.3, .7], [.01, .99]]
        #log_loss(pred, prob)
