import numpy as np
import csv
import tflearn
import tensorflow as tf
from sklearn.metrics import accuracy_score

def load_training_data(filename):
  data, labels = [], []
  with open(filename) as dataset_csv:
    reader = csv.reader(dataset_csv)
    next(reader, None) # skip header
    # extract feature vectors and labels
    for row in reader:
      labels.append(row.pop(-1))
      data.append(np.asarray(row, np.float))
    data = np.array(data)
    labels = np.array(labels, dtype=np.int)
    return (data, labels)

def load_input_data(filename):
  data, tags = [], []
  with open(filename) as dataset_csv:
    reader = csv.reader(dataset_csv)
    next(reader, None) # skip header
    # extract feature vectors and tags
    for row in reader:
      tags.append(row.pop(0))
      data.append(np.asarray(row, np.float))
    return (data, tags)

# process input data
t_data, t_labels = load_training_data('numerai_training_data.csv')
p_data, p_tags = load_input_data('numerai_tournament_data.csv')

# create training and validation sets
split_idx = int(0.8 * len(t_data))
tt_data, tv_data = t_data[:split_idx], t_data[split_idx:]
tt_labels, tv_labels = t_labels[:split_idx], t_labels[split_idx:]

tt_data = np.reshape(tt_data, (-1, 21))
tt_labels = np.reshape(tt_labels, (-1, 1))
tv_data = np.reshape(tv_data, (-1,21))
tv_labels = np.reshape(tv_labels, (-1,1))

assert len(tt_data) == len(tt_labels), "# feature vectors != # labels"
print("# Labels: {0}".format(len(tt_labels)))
print("len feature vector: {0}".format(len(tt_data[0])))

print("First feature vector\n{0}".format(tt_data[0]))
print("First label: {0}".format(tt_labels[0]))
print(len(tv_data))
print(len(tv_labels))

# input layer takes an unspecified amount of 21-vectors
input_layer = tflearn.input_data(shape=[None, 21], name="inputlayer")

dense1 = tflearn.fully_connected(input_layer, 12, activation='sigmoid',
                                         regularizer='L2', weight_decay=0.001)
dropout = tflearn.dropout(dense1, 0.4)
dense2 = tflearn.fully_connected(dropout, 12, activation='sigmoid',
                                         regularizer='L2', weight_decay=0.001)
softmax = tflearn.fully_connected(dense2, 1, activation='softmax')

sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
net = tflearn.regression(softmax, optimizer=sgd,
                                 loss='mean_square')

model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(tt_data, tt_labels, n_epoch=10, validation_set=(tv_data, tv_labels),
                  show_metric=True, run_id="dense_model")
predictions = model.predict(tv_data)
print(len(predictions))
print(predictions)
print(accuracy_score(tv_labels, predictions))

# simple layer with a sigmoid activation function and 10 nodes
# fc1 = tflearn.fully_connected(inputlayer, 10, activation='sigmoid', name='fc1')
# fc1 = tflearn.fully_connected(inputlayer, 1, activation='sigmoid', name='fc1')

# Regression using SGD with learning rate decay and Top-3 accuracy
# stochastic gradient descent optimizer
# sgd = tflearn.SGD(learning_rate=0.001, lr_decay=0.96, decay_step=1000, name='sgd')
# 
# # network
# net = tflearn.regression(fc1, optimizer=sgd, loss='mean_square', name='lmao')
# 
# # create model
# model = tflearn.DNN(net, tensorboard_verbose=0)
# model.fit(tt_data, tt_labels, n_epoch=5, show_metric=True, validation_set=(tv_data, tv_labels))
# 
# # input_layer = tflearn.input_data(shape=[None, 21])
# print(len(tv_data))
# predictions = model.predict(tv_data)
# print(predictions)
# results = list(map(lambda x: round(x[0]), model.predict(tv_data)))
# accuracy_score(tv_labels, results)


# print(ret)

# result = model.predict(tv_data)
# print(result)

# accuracy_score = model.evaluate(tv_data, y=tv_labels)["accuracy"]
# print(accuracy_score)

# input_layer = tflearn.input_data(shape=[None, 21])
# dense1 = tflearn.fully_connected(input_layer, 64, activation='tanh',
#                                          regularizer='L2', weight_decay=0.001)
# dropout1 = tflearn.dropout(dense1, 0.8)
# dense2 = tflearn.fully_connected(dropout1, 64, activation='tanh',
#                                          regularizer='L2', weight_decay=0.001)
# dropout2 = tflearn.dropout(dense2, 0.8)
# softmax = tflearn.fully_connected(dropout2, 1, activation='softmax')
# 
# # Regression using SGD with learning rate decay and Top-3 accuracy
# sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
# top_k = tflearn.metrics.Top_k(3)
# net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
#                                  loss='categorical_crossentropy')
# 
# model = tflearn.DNN(net, tensorboard_verbose=0)
# model.fit(tt_data, tt_labels, n_epoch=20, validation_set=(tv_data, tv_labels),
#                   show_metric=True, run_id="dense_model")




