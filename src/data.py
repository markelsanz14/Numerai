#!/usr/bin/python3

# sanitization routines for dataset
import csv
import sys

def getdata(filename):
    trainingset = []
    with(open(filename) as dataset:
        reader = csv.reader(dataset)
        trainingset = [(row[:-1], row[-1]) for row in reader]
    del trainingset[0]
    return trainingset
