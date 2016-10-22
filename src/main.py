from sys import argv
from data import *
from models import *

available_models = {
        'svm': SvmModel,
        'gaussiannb': GaussianNBModel
}

usage = '''
Usage:
    main.py help
    main.py crossvalidate <model> <training.csv>
    main.py predict <model> <training.csv> <predict.csv>
'''

crossval_percentage = 0.8

def showhelp():
    print(usage)
    print("Available models:")
    print(available_models)

if __name__ == "__main__":
    # validate usage
    if len(argv) < 2 or argv[1] == "help":
        showhelp()
        exit()

    # get dataset
    training_data = as_dataset(argv[3])
    # get model
    model = available_models[argv[2]]()

    if argv[1] == "crossvalidate":
        score = model.crossvalidate(training_data, crossval_percentage)
        print("Accuracy of {0}: {1}".format(argv[2], score))
    elif argv[1] == "predict":
        prediction_data = as_predictionset(argv[4])
        model.fit(training_data)
        predictions = model.predict(prediction_data)
