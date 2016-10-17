#!/usr/bin/python
import sys
from data import process_dataset

usage = """
Usage: ./main.py <training.csv>

"""

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(usage)
        exit()
    trainingfile = sys.argv[1]
    dataset = process_dataset(trainingfile)
    mysvm = svm(dataset)
    mysvm.fit()
