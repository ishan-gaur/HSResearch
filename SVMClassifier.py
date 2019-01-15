import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


# important constants for training
step_size = 0.000001  # for step size
Epochs = 10000  # number of times the computer preforms a weight update
num_examples = 68106
num_features = 216
num_labels = 1

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
        Y[i][0] = row[222]
        i += 1
        if i == num_examples:
        	break

f.close

half = (int(len(X) / 2) + 1)
x_train = np.array(X[:half][:])
y_train = np.reshape(np.array(Y[:half][:]), (-1))
x_test = np.array(X[half:][:])
y_test = np.reshape(np.array(Y[half:][:]), (-1))

# Training the Nu SVR model
clf = SVC()
clf.fit(x_train, y_train)

# Gathering predictions from the model
ytrain = clf.predict(x_train)
ytest = clf.predict(x_test)

# Print performance metrics
print('---------Training-------')
print('Explained Variance Score', explained_variance_score(y_train, ytrain), 'Out of 1.00')
print('Mean Absolute Error', mean_absolute_error(y_train, ytrain))
print('Mean Squared Error', mean_squared_error(y_train, y_train))
print('Median Absolute Error', median_absolute_error(y_train, ytrain))
print('R2 Score', r2_score(y_train, ytrain), 'Out of 1.00')
print('Average Percent Error', (mean_absolute_error(y_train, ytrain)/np.average(y_train)))

print('---------Testing-------')
print('Explained Variance Score', explained_variance_score(y_test, ytest), 'Out of 1.00')
print('Mean Absolute Error', mean_absolute_error(y_test, ytest))
print('Mean Squared Error', mean_squared_error(y_test, y_test))
print('Median Absolute Error', median_absolute_error(y_test, ytest))
print('R2 Score', r2_score(y_test, ytest), 'Out of 1.00')
print('Average Percent Error', (mean_absolute_error(y_test, ytest)/np.average(y_test)))