import csv
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug


# important constants for training
step_size = 0.00001  # for step size
Epochs = 1000  # number of times the computer preforms a weight update
num_features = 216
num_examples = 68000
num_neurons = 100
num_labels = 2

# Setting up the training set using a csv file
f = open('NACCDATAFINAL.csv')
csv_r = csv.reader(f)

i = 0
X = np.zeros((num_examples, num_features))  # this will contain all of the X vectors for the training data
Y = np.zeros((num_examples, num_labels))  # this will contain all of the desired outputs from the training data

# Used to assign all of the data values from the csv file to the numpy arrays

for row in csv_r:
    if row[1] != "NACCID":
        X[i] = row[4:220]
        if row[222] == '0':
            Y[i][0] = 1
        else:
            Y[i][1] = 1
        i += 1
        if i == num_examples:
            break

f.close

# Get data and separate into testing and training data
half = (int(len(X) / 2) + 1)
x_train = X[:half][:]
y_train = Y[:half][:]
x_test = X[half:][:]
y_test = Y[half:][:]

with tf.name_scope("inputs"):
    x = tf.placeholder(tf.float32, [None, num_features], name="inputs")

# Setup of forward propagation
with tf.name_scope("hidden_layer1"):
    weights_ih1 = tf.Variable(tf.truncated_normal([num_features, num_neurons]))
    b_h1 = tf.Variable(tf.truncated_normal([num_neurons])) 
    hidden1 = tf.nn.elu(tf.matmul(x, weights_ih1) + b_h1)
    tf.summary.histogram("hidden_layer1_weights", weights_ih1)
    tf.summary.histogram("hidden_layer1_biases", b_h1)

with tf.name_scope("hidden_layer2"):
    weights_h1h2 = tf.Variable(tf.truncated_normal([num_neurons, num_neurons]))
    b_h2 = tf.Variable(tf.truncated_normal([num_neurons]))
    hidden2 = tf.nn.elu(tf.matmul(hidden1, weights_h1h2) + b_h2)
    tf.summary.histogram("hidden_layer2_weights", weights_h1h2)
    tf.summary.histogram("hidden_layer2_biases", b_h2)

with tf.name_scope("hidden_layer3"):
    weights_h2h3 = tf.Variable(tf.truncated_normal([num_neurons, num_neurons]))
    b_h3 = tf.Variable(tf.truncated_normal([num_neurons]))
    hidden3 = tf.nn.elu(tf.matmul(hidden2, weights_h2h3) + b_h3)
    tf.summary.histogram("hidden_layer3_weights", weights_h2h3)
    tf.summary.histogram("hidden_layer3_biases", b_h3)

with tf.name_scope("output_layer"):
    weights_h3o = tf.Variable(tf.truncated_normal([num_neurons, num_labels]))
    b_o = tf.Variable(tf.truncated_normal([num_labels]))
    #y_ = tf.nn.softmax(tf.matmul(hidden1, weights_h1o) + b_o, name="outputs")
    y_ = tf.add(tf.matmul(hidden3, weights_h3o), b_o, name="outputs")
    tf.summary.histogram("output_weights", weights_h3o)
    tf.summary.histogram("output_biases", b_o)

# Setup for training
with tf.name_scope("loss"):
    y = tf.placeholder(tf.float32, [None, num_labels], name="labels")
    prediction = tf.nn.softmax(y_)
    #loss = tf.nn.l2_loss(y - prediction)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
    tf.summary.histogram("labels", y)
    tf.summary.histogram("prediction", prediction)
    tf.summary.histogram("loss", loss)

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(step_size).minimize(loss)

with tf.name_scope("accuracy"):
    #correct_prediction = tf.equal(tf.round(y), tf.round(prediction))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

# Implementation of training
init = tf.initialize_all_variables()
#sess = tf_debug.LocalCLIDebugWrapperSession(tf.Session())
sess = tf.Session()
sess.run(init)
"""
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("/tmp/alz_experiments/27")
writer.add_graph(sess.graph)
"""
for i in range(Epochs):
    start = int(i * (num_examples / 5))
    end = int((i + 1) * (num_examples / 5))
    sess.run(train_step, feed_dict={x: x_train[start:end], y: y_train[start:end]})
    print(sess.run(y_, feed_dict={x: x_train[start:end], y: y_train[start:end]}))
    """
    if (i + 1) % 1 == 0:
        summary = sess.run(merged_summary, feed_dict={x: x_train, y: y_train})
        writer.add_summary(summary, i)
    """
    if (i + 1) % 100 == 0:
        print("----------------------------------------------------")
        print("Training Accuracy")
        print(sess.run(accuracy, feed_dict={x: x_train, y: y_train}))
        print("Testing Accuracy")
        print(sess.run(accuracy, feed_dict={x: x_test, y: y_test}))
        #print("----------------------------------------------------")
#print(sess.run(y_, feed_dict={x: x_train, y: y_train}))
