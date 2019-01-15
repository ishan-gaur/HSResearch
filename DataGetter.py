import csv
import numpy as np
import tensorflow as tf


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
count = 0
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
            count += 1
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

print(count / (num_examples - count))