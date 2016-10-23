# utilities
import sys
import csv
from sys import argv
# from models import *
import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
# classifiers
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegressionCV

# Configuration parameters
crossval_percentage = 0.8


# Available classifiers to choose from
classifiers = {
    'svm': SVC(),
    'gaussiannb': GaussianNB(),
    'mlp': MLPClassifier(),
    'mlpr': MLPRegressor(solver='sgd'),
    'lr': LogisticRegressionCV()
}

# Data processing
def process_training_csv(filename):
    """
    Returns a list of tuples:
     ([f1..fn], class)
    """
    dataset = None
    with open(filename) as dataset_csv:
        reader = csv.reader(dataset_csv)
        next(reader, None) # skip header
        dataset = [([float(x) for x in row[:-1]], int(row[-1])) for row in reader]
    return dataset

def process_input_csv(filename):
    '''
    Returns a list of feature vectors:
     [f1..fn]
    '''
    dataset = None
    with open(filename) as dataset_csv:
        reader = csv.reader(dataset_csv)
        next(reader, None) # skip header
        dataset = [ (row[0], list(map(lambda x: float(x), row[1:]))) for row in reader ]
    return dataset

# fitting function for general model
def fit(features, labels, classifier):
    '''
    Fit the model to a set of data.
    training_data: [([feature_vec], label)]
    '''
    classifier.fit(features, labels)

def predict(features, classifier):
    '''
    Generate predictions using the fitted classifier
    '''
    return classifier.predict(features)


# User interface
usage = '''
Usage:
    main.py help
    main.py classify <classifier> <training.csv> <input.csv>
    main.py crossvalidate <classifier> <training.csv>
'''

def showhelp():
    print(usage)
    print("Available classifiers:")
    print(classifiers.keys())

if __name__ == "__main__":
    # validate usage
    if len(argv) < 2 or argv[1] == "help":
        showhelp()
        exit()

    classifier_name = argv[2]
    training_filename = argv[3]
    input_filename = argv[4] if len(argv) > 4 else None

    # build training dataset
    t_data      = process_training_csv(training_filename)
    t_features  = np.array([entry[0] for entry in t_data])
    t_labels    = np.array([entry[1] for entry in t_data])

    # get classifier
    classifier = classifiers[classifier_name]

    if argv[1] == "crossvalidate":
        # split the data
        split_idx = int(crossval_percentage * len(t_data))
        # split training data into training and validation data
        cv_t_features = t_features[:split_idx]
        cv_t_labels   = t_labels[:split_idx]
        cv_p_features = t_features[split_idx:]
        cv_p_labels   = t_labels[split_idx:]

        assert len(cv_t_features) == len(cv_t_labels), "Feature length inequal."
        print("Training set size: {0}".format(len(cv_t_features)))
        print("Validation set size: {0}".format(len(cv_p_features)))

        # train with the training part
        print("Fitting...")
        fit(cv_t_features, cv_t_labels, classifier)

        # make predictions on the other part
        print("Classifying...")
        predictions = predict(cv_p_features, classifier)

        # normalize predictions to classes
        normalized = list(map(lambda x: round(x), predictions))
        score = accuracy_score(normalized, cv_p_labels)
        print("Accuracy of {0}: {1}".format(classifier_name, score))
        if hasattr(classifier, 'loss_'):
            print("Log-loss: {0}".format(classifier.loss_))

    elif argv[1] == "classify":
        p_data = process_input_csv(input_filename)
        p_tags = np.array([entry[0] for entry in p_data])
        print(p_tags)
        p_features = np.array([entry[1] for entry in p_data])

        print("Training set size: {0}".format(len(t_features)))

        # train
        print("Fitting...")
        fit(t_features, t_labels, classifier)

        # predict
        print("Classifying...")
        predictions = predict(p_features, classifier)

        print('"t_id","probability"')
        # dump predictions
        results = list(zip(p_tags, predictions))
        for pair in results:
            print("{0},{1}".format(pair[0], pair[1]))
