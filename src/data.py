# sanitization routines for dataset
import csv
import sys

def as_dataset(filename):
    """
    Returns a list of tuples:
    [
      ([f1,f2,f3,..], class),
      ([f1,f2,f3,..], class),
      ...
    ]
    """
    trainingset = []
    with open(filename) as dataset:
        reader = csv.reader(dataset)
        next(reader, None) # skip header
        trainingset = [([float(x) for x in row[:-1]], int(row[-1])) for row in reader]
    return trainingset

def as_predictionset(filename):
    predictionset = []
    with open(filename) as dataset:
        reader = csv.reader(dataset)
        next(reader, None) # skip header
        predictionset = [ map(lambda x: float(x), row) for row in reader ]
    return predictionset
