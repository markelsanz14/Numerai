#!/usr/bin/python
import sys
from data import *
from svm import *
from visualizeData import *

usage = 'Usage: ./main.py <training.csv>'

def crossvalidate(dataset):
    # split the data 80/20
    split_idx = int(.8 * len(dataset))
    trainingset = dataset[:split_idx]
    validateset = dataset[split_idx:]

    print ("# items in training set: {0}".format(len(trainingset)))
    print ("# items in test set: {0}".format(len(validateset)))

    # make a new svm for this dataset
    xvalsvm = SvmModel()
    print ("Fitting model...")
    xvalsvm.fit(trainingset)
    print ("Calculating predictions...")
    print ("Accuracy: {0}".format(xvalsvm.predict(validateset)))

def visualize(dataset):
	vis = visualizeData(dataset)
	#vis.visualizeAll()

if __name__ == "__main__":
    # validate usage
    if len(sys.argv) < 2:
        print(usage)
        exit()

    # get training data
    trainingfile = sys.argv[1]
    dataset = as_dataset(trainingfile)

    # crossvalidate
    #print ("Performing crossvalidation...")
    #crossvalidate(dataset)

    visualize(dataset)

    # predictions
    # ...

