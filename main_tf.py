# utilities
import sys
import csv
from sys import argv
# from models import *
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


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


if __name__ == "__main__":

    training_filename = argv[1]
    # build training dataset
    t_data      = process_training_csv(training_filename)
    t_features  = np.array([entry[0] for entry in t_data])
    t_labels    = np.array([entry[1] for entry in t_data])


    num_features = 21
    num_labels = 2


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, num_features)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
t_features, t_labels = reformat(t_features, t_labels)
#valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)


#TensorFlow works like this:
#First you describe the computation that you want to see performed: what the inputs, the variables, and the operations look like. These get created as nodes over a computation graph. This description is all contained within the block below:
#with graph.as_default():
#    ...
#Then you can run the operations on this graph as many times as you want by calling session.run(), providing it outputs to fetch from the graph that get returned. This runtime operation is all contained in the block below:
#with tf.Session(graph=graph) as session:
#    ...
#Let's load all the data into TensorFlow and build the computation graph corresponding to our training:

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 10000

graph = tf.Graph()
with graph.as_default():
    # Input data.
    # Load the training, validation and test data into constants that are
    # attached to the graph.
    tf_train_dataset = tf.constant(t_features[:,:])
    tf_train_labels = tf.constant(t_labels[:])
    #tf_valid_dataset = tf.constant(valid_dataset)
    #tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    # These are the parameters that we are going to be training. The weight
    # matrix will be initialized using random values following a (truncated)
    # normal distribution. The biases get initialized to zero.
    weights = tf.Variable(tf.truncated_normal([num_features, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    # We multiply the inputs with the weight matrix, and add biases. We compute
    # the softmax and cross-entropy (it's one operation in TensorFlow, because
    # it's very common, and it can be optimized). We take the average of this
    # cross-entropy across all training examples: that's our loss.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    # We are going to find the minimum of this loss using gradient descent.
    # Learning rate = 0.5
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    # These are not part of training, but merely here so that we can report
    # accuracy figures as we train.
    train_prediction = tf.nn.softmax(logits)
    #valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    #test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

num_steps = 801

def accuracy(predictions, labels):
    #a = np.argmax(predictions,1)
    #b = np.argmax(labels,1)
    #c = np.sum(a == b)
    #d = 100 * c
    #e = d / predictions.shape[0]
    return (100.0 * np.sum(np.argmax(predictions) == np.argmax(labels)) / predictions.shape[0])

with tf.Session(graph=graph) as session:
    # This is a one-time operation which ensures the parameters get initialized as
    # we described in the graph: random weights for the matrix, zeros for the
    # biases. 
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        # Run the computations. We tell .run() that we want to run the optimizer,
        # and get the loss value and the training predictions returned as numpy
        # arrays.
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        if (step % 100 == 0):
            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy: %.1f%%' % accuracy(predictions, tf_train_labels))
            # Calling .eval() on valid_prediction is basically like calling run(), but
            # just to get that one numpy array. Note that it recomputes all its graph
            # dependencies.
            #print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
    #print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

