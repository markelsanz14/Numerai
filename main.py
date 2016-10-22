import sys
import csv
from sys import argv
from models import *

# Configuration parameters
crossval_percentage = 0.8


# Available classifiers to choose from
classifiers = {
    'svm': SvmModel,
    'gaussiannb': GaussianNBModel,
    'mlp': MLPCModel,
    'lr': LogisticRegressionModel
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
        dataset = [ map(lambda x: float(x), row) for row in reader ]
    return predictionset


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
    training_data = process_training_csv(training_filename)

    # get classifier
    classifier = classifiers[classifier_name]()

    if argv[1] == "crossvalidate":
        score = classifier.crossvalidate(training_data, crossval_percentage)
        print("Accuracy of {0}: {1}".format(classifier_name, score))

    elif argv[1] == "classify":
        prediction_data = process_input_csv(input_filename)
        classifier.fit(training_data)
        predictions = classifier.classify(prediction_data)
