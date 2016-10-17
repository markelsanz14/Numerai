from sklearn import svm
from sklearn.metrics import log_loss

class svm:

    def __init__(self, training_set):
        self.training_set = training_set
        self.classifier = svm.SVC()
        features = [entry[0] for entry in self.training_set]
        labels = [entry[1] for entry in self.training_set]
        self.classifier.fit(features, labels)

    def predict():
        pred = self.classifier.predict()
        dist = self.classifier.decision_function()
        max_pos = max(dist)
        max_neg = min(dist)
        prob = map(lambda x: x/max_neg if x < 0 else x/max_pos, dist)
        #y_true = [0, 0, 1, 1]
        #y_pred = [[.9, .1], [.8, .2], [.3, .7], [.01, .99]]
        #log_loss(pred, prob)
