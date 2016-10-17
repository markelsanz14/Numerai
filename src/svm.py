from sklearn import svm
from sklearn.metrics import log_loss

class svm:

    def __init__(self, training_set):
        self.training_set = training_set
        self.classifier = svm.SVC()
        self.classifier.fit(self.training_set[features], selft.training_set[labels])

    def predict():
        pred = self.classifier.predict()
        dist = self.classigier.decision_function()
        #y_true = [0, 0, 1, 1]
        #y_pred = [[.9, .1], [.8, .2], [.3, .7], [.01, .99]]
        #log_loss(training_set[labels], y_pred)
