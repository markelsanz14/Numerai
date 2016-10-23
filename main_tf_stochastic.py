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

batch_size = 128

graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_features))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    #tf_valid_dataset = tf.constant(valid_dataset)
    #tf_test_dataset = tf.constant(test_dataset)

   # Variables.
    weights = tf.Variable(tf.truncated_normal([num_features, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    #valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    #test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


num_steps = 123001

def accuracy(predictions, labels):
    a = np.argmax(predictions)
    b = np.argmax(labels)
    c = np.sum(a == b)
    print a
    print b
    print c
    d = 100 * c
    e = d / predictions.shape[0]
    return (100.0 * np.sum(np.argmax(predictions,0) == np.argmax(labels,0)) / predictions.shape[0])

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (t_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = t_features[offset:(offset + batch_size), :]
        batch_labels = t_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            #print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            #print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
    #print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

