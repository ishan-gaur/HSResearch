"""Dynamic asset allocation using an DNN and St. Louis Fed Bank Data."""
import tensorflow as tf
import sklearn.preprocessing as preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Getting the data
STOCKS_TRAIN = "/Users/IshanGaur/Dropbox/Ben/Brain/StocksData/csvs/StocksTrain2.csv"
STOCKS_TEST = "/Users/IshanGaur/Dropbox/Ben/Brain/StocksData/csvs/StocksTest2.csv"

df_train = pd.read_csv(STOCKS_TRAIN).fillna(0)
df_test = pd.read_csv(STOCKS_TEST).fillna(0)

train_data = df_train.as_matrix().astype(float)
test_data = df_test.as_matrix().astype(float)

data = np.append(train_data, test_data, axis=0)
data = preprocessing.normalize(data)
data = preprocessing.MinMaxScaler().fit_transform(data)
print(data)

train_data = data[:-100, :]
test_data = data[-100:, :]

np.random.shuffle(train_data)
np.random.shuffle(test_data)

x_train = train_data[:, 1:]
y_train = train_data[:, 0]
x_test = test_data[:, 1:]
y_test = test_data[:, 0]

# Constants
num_outputs = 1
num_features = train_data.shape[1] - num_outputs

step_size = tf.constant(10**-15, dtype='float64')
num_epochs = 100

num_neurons = [60, 70, 110, 80, 100, 130, 90, 150, 110, 50, 30]

# Reshape data
x_train = np.reshape(x_train, [-1, num_features])
y_train = np.reshape(y_train, [-1, num_outputs])
x_test = np.reshape(x_test, [-1, num_features])
y_test = np.reshape(y_test, [-1, num_outputs])

# Setup the architecture of the neural net
with tf.name_scope('Input_Layer') as scope:
    x = tf.placeholder(dtype='float64', shape=[None, num_features], name='Inputs')
    y = tf.placeholder(dtype='float64', shape=[None, num_outputs], name='Labels')

    weights_ih = tf.Variable(initial_value=tf.truncated_normal([num_features, num_neurons[0]], dtype='float64'), name='Input_Layer_Weights', dtype='float64')
    b_ih = tf.Variable(tf.truncated_normal([num_neurons[0]], dtype='float64'), name='Input_Layer_Biases', dtype='float64')
    input_layer = tf.nn.elu(tf.matmul(x, weights_ih) + b_ih)

with tf.name_scope('Hidden_Layers') as scope:
    with tf.name_scope('Layer_1') as scope:
        weights_hh1 = tf.Variable(tf.truncated_normal([num_neurons[0], num_neurons[1]], dtype='float64'), name='First_Layer_Weights', dtype='float64')
        b_hh1 = tf.Variable(tf.truncated_normal([num_neurons[1]], dtype='float64'), name='First_Layer_Biases', dtype='float64')
        hidden1 = tf.nn.relu(tf.matmul(input_layer, weights_hh1) + b_hh1)

    with tf.name_scope('Layer_2') as scope:
        weights_hh2 = tf.Variable(tf.truncated_normal([num_neurons[1], num_neurons[2]], dtype='float64'), name='Second_Layer_Weights', dtype='float64')
        b_hh2 = tf.Variable(tf.truncated_normal([num_neurons[2]], dtype='float64'), name='Second_Layer_Biases', dtype='float64')
        hidden2 = tf.nn.softplus(tf.matmul(hidden1, weights_hh2) + b_hh2)

    with tf.name_scope('Layer_3') as scope:
        weights_hh3 = tf.Variable(tf.truncated_normal([num_neurons[2], num_neurons[3]], dtype='float64'), name='Third_Layer_Weights', dtype='float64')
        b_hh3 = tf.Variable(tf.truncated_normal([num_neurons[3]], dtype='float64'), name='Third_Layer_Biases', dtype='float64')
        hidden3 = tf.nn.elu(tf.matmul(hidden2, weights_hh3) + b_hh3)

    with tf.name_scope('Layer_4') as scope:
        weights_hh4 = tf.Variable(tf.truncated_normal([num_neurons[3], num_neurons[4]], dtype='float64'), name='Fourth_Layer_Weights', dtype='float64')
        b_hh4 = tf.Variable(tf.truncated_normal([num_neurons[4]], dtype='float64'), name='Fourth_Layer_Biases', dtype='float64')
        hidden4 = tf.nn.softmax(tf.matmul(hidden3, weights_hh4) + b_hh4)

    with tf.name_scope('Layer_5') as scope:
        weights_hh5 = tf.Variable(tf.truncated_normal([num_neurons[4], num_neurons[5]], dtype='float64'), name='Fifth_Layer_Weights', dtype='float64')
        b_hh5 = tf.Variable(tf.truncated_normal([num_neurons[5]], dtype='float64'), name='Fifth_Layer_Biases', dtype='float64')
        hidden5 = tf.nn.sigmoid(tf.matmul(hidden4, weights_hh5) + b_hh5)

    with tf.name_scope('Layer_6') as scope:
        weights_hh6 = tf.Variable(tf.truncated_normal([num_neurons[5], num_neurons[6]], dtype='float64'), name='Sixth_Layer_Weights', dtype='float64')
        b_hh6 = tf.Variable(tf.truncated_normal([num_neurons[6]], dtype='float64'), name='Sixth_Layer_Biases', dtype='float64')
        hidden6 = tf.nn.tanh(tf.matmul(hidden5, weights_hh6) + b_hh6)

    with tf.name_scope('Layer_7') as scope:
        weights_hh7 = tf.Variable(tf.truncated_normal([num_neurons[6], num_neurons[7]], dtype='float64'), name='Seventh_Layer_Weights', dtype='float64')
        b_hh7 = tf.Variable(tf.truncated_normal([num_neurons[7]], dtype='float64'), name='Seventh_Layer_Biases', dtype='float64')
        hidden7 = tf.nn.elu(tf.matmul(hidden6, weights_hh7) + b_hh7)

    with tf.name_scope('Layer_8') as scope:
        weights_hh8 = tf.Variable(tf.truncated_normal([num_neurons[7], num_neurons[8]], dtype='float64'), name='Eighth_Layer_Weights', dtype='float64')
        b_hh8 = tf.Variable(tf.truncated_normal([num_neurons[8]], dtype='float64'), name='Eighth_Layer_Biases', dtype='float64')
        hidden8 = tf.nn.tanh(tf.matmul(hidden7, weights_hh8) + b_hh8)

    with tf.name_scope('Layer_9') as scope:
        weights_hh9 = tf.Variable(tf.truncated_normal([num_neurons[8], num_neurons[9]], dtype='float64'), name='Ninth_Layer_Weights', dtype='float64')
        b_hh9 = tf.Variable(tf.truncated_normal([num_neurons[9]], dtype='float64'), name='Ninth_Layer_Biases', dtype='float64')
        hidden9 = tf.nn.softplus(tf.matmul(hidden8, weights_hh9) + b_hh9)

    with tf.name_scope('Layer_10') as scope:
        weights_hh10 = tf.Variable(tf.truncated_normal([num_neurons[9], num_neurons[10]], dtype='float64'), name='Tenth_Layer_Weights', dtype='float64')
        b_hh10 = tf.Variable(tf.truncated_normal([num_neurons[10]], dtype='float64'), name='Tenth_Layer_Biases', dtype='float64')
        hidden10 = tf.nn.elu(tf.matmul(hidden9, weights_hh10) + b_hh10)


with tf.name_scope('Output_Layer') as scope:
    weights_ho = tf.Variable(tf.truncated_normal([num_neurons[10], num_outputs], dtype='float64'), name='Output_Layer_Weights', dtype='float64')
    b_ho = tf.Variable(tf.truncated_normal([num_outputs], dtype='float64'), name='Output_Layer_Biases', dtype='float64')

    y_ = tf.nn.tanh(tf.matmul(hidden10, weights_ho) + b_ho)

# Setup for training
l2_loss = tf.nn.l2_loss((y - y_))
optimize = tf.train.GradientDescentOptimizer(step_size).minimize(l2_loss)
error = tf.reduce_sum((y_ - y))
tf.summary.scalar("Percent_Error", error)

# Implementation of training
init = tf.global_variables_initializer()
sess = tf.Session()
file_writer = tf.summary.FileWriter('/Users/IshanGaur/Dropbox/Ben/Brain/Checkpoints/DNN/', sess.graph)
sess.run(init)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

training_record = []
testing_record = []
for i in range(num_epochs):
    sess.run(optimize, feed_dict={x: x_train, y: y_train})
    loss = sess.run(l2_loss, feed_dict={x: x_train, y: y_train})
    _error = sess.run(error, feed_dict={x: x_train, y: y_train})
    print('For Epoch', i, 'the Loss is:', loss, 'The Error is:', _error)

    if (i + 1) % 50 == 0:
        taa = sess.run(error, feed_dict={x: x_train, y: y_train})
        tae = sess.run(error, feed_dict={x: x_test, y: y_test})
        training_record.append(taa)
        testing_record.append(tae)
        values = sess.run(y_, feed_dict={x: x_train})
        print("----------------------------------------------------")
        print("Training Accuracy")
        print(taa)
        print("Testing Accuracy")
        print(sess.run(error, feed_dict={x: x_test, y: y_test}))
        print("----------------------------------------------------")

print(sess.run(y_, feed_dict={x: x_test, y: y_test}))
print(y_test)
print("---------------------------")
print("---------------------------")
print("---------------------------")
print(sess.run(y_, feed_dict={x: x_train, y: y_train}))
print(y_train)
