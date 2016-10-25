import numpy as np
import csv
import tflearn
import tensorflow as tf

def load_training_data(filename):
  data, labels = [], []
  with open(filename) as dataset_csv:
    reader = csv.reader(dataset_csv)
    next(reader, None) # skip header

    # extract feature vectors and labels
    for row in reader:
      labels.append(row.pop(-1))
      data.append(np.asarray(row, dtype=np.float128))
    data = np.array(data)

    labels = np.array(labels, dtype=np.int)
    labels = tflearn.data_utils.to_categorical(labels, 2)
    return (data, labels)

def load_input_data(filename):
  data, tags = [], []
  with open(filename) as dataset_csv:
    reader = csv.reader(dataset_csv)
    next(reader, None) # skip header
    # extract feature vectors and tags
    for row in reader:
      tags.append(row.pop(0))
      data.append(np.asarray(row, np.float128))
    return (data, tags)

# process input data
t_data, t_labels = load_training_data('numerai_training_data.csv')

print("t_data[0][0]: {0}\nt_data[0]: {1}".format(t_data[0][0], t_data[0]))

# input layer takes an unspecified amount of 21-vectors
net = tflearn.input_data(shape=[None, 21])
net = tflearn.fully_connected(net, 8, activation='relu')
net = tflearn.fully_connected(net, 2, activation='softmax')

adadelta = tf.train.AdadeltaOptimizer()
net = tflearn.regression(net, optimizer=adadelta, loss='softmax_categorical_crossentropy')

model = tflearn.DNN(net)
model.fit(t_data, t_labels, n_epoch=200000, batch_size=256, show_metric=True, validation_set=0.15, snapshot_step=10000, snapshot_epoch=False)

with open("model.tf", 'w') as model_file:
  model.save(model_file)
  model_file.close()
