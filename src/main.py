#!/usr/bin/python
import sys
from data import *
from svm import *

usage = 'Usage: ./main.py <training.csv>'

def crossvalidate(dataset):
    # split the data 80/20
    dataset = dataset[:int(.25 * len(dataset))]
    print ("Length of dataset: {0}".format(len(dataset)))

    split_idx = int(.8 * len(dataset))
    trainingset = dataset[:split_idx]
    validateset = dataset[-split_idx:]

    # make a new svm for this dataset
    xvalsvm = SvmModel()
    print ("Fitting model...")
    xvalsvm.fit(trainingset)
    print ("Calculating predictions...")
    print (xvalsvm.predict(validateset))


if __name__ == "__main__":
    # validate usage
    if len(sys.argv) < 2:
        print(usage)
        exit()

    # get training data
    trainingfile = sys.argv[1]
    dataset = as_dataset(trainingfile)

    # crossvalidate
    print ("Performing crossvalidation...")
    crossvalidate(dataset)

    # predictions
    # ...

