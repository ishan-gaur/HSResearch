"""Nu Support Vector Regression """
print('Importing libraries...')
from sklearn.svm import NuSVR
from sklearn.preprocessing import normalize, minmax_scale
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

start = 9960
end = 10715
smooth = 60

# Getting the data
print('Getting data...')
STOCKS_DATA = "/Users/IshanGaur/Dropbox/Ben/Brain/StocksData/csvs/StocksData3.csv"
df = pd.read_csv(STOCKS_DATA).fillna(0)

# Preprocessing
print('Preprocessing the data...')
initial_data = df.as_matrix().astype(float)
norm_data = normalize(initial_data, axis=0)
data = minmax_scale(norm_data)

# Selecting training data and a testing time period
train_data = np.append(data[:start, :], data[end:], axis=0)
test_data = data[start:end, :]

# Shuffling the data to make sure it is learned in no particular order
np.random.shuffle(train_data)
np.random.shuffle(test_data)

# Separating the data into features and targets
print('Forming training and testing sets...')
x_train = train_data[:, 1:]
y_train = train_data[:, 0]
x_test = test_data[:, 1:]
y_test = test_data[:, 0]
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Reshape data to meet the requirements of the modle
y_train = np.reshape(y_train, [-1])
y_test = np.reshape(y_test, [-1])


# Training the Nu SVR model
print('Building and training the Nu SVR model...')
clf = NuSVR(kernel='poly', gamma=0.0523125)
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

# Reversing the effects of preprocessing on the prediction
labels = np.reshape(data[start:end, 0], [-1])
#w = np.sqrt(sum(np.power(initial_data[start:end, 0], 2)))

initial_pred = np.reshape(clf.predict(data[start:end, 1:]), [-1])
unscaled_pred = initial_pred * (np.amax(labels) - np.amin(labels)) + np.amin(labels)
#true_pred = unscaled_pred * w

real_value = np.reshape(initial_data[start:end, 0], [-1])

#print('-----------------------')
#print(np.reshape(data[start:start + 10, 0], [-1]) - initial_pred[:10])
#print(np.reshape(norm_data[start:start + 10, 0], [-1]) - unscaled_pred[:10])
#print(np.reshape(initial_data[start:start + 10, 0], [-1]) - true_pred[:10])
#print('-----------------------')
#print(np.reshape(data[start:start + 10, 0], [-1]))
#print(np.reshape(norm_data[start:start + 10, 0], [-1]))
#print(np.reshape(initial_data[start:start + 10, 0], [-1]))
#print('-----------------------')


def movingaverage(interval, window_size):
    """Return a widow_size moving average of the given interval."""
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

# Sample testing periods
# 8451:8700 - 2001(start):2001(end)
# 9456:10967 - 2005(start):2010(end)
# 9960:10715 - 2007(start):2009(end)

# Plot the prediction vs the actual data
fig = plt.figure()

fig.suptitle('NYSE Closing Price Jan 2007 to Dec 2009', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)

ax.set_xlabel('Day Since Start of Time Period')
ax.set_ylabel('Closing Price')

plt.plot(movingaverage(unscaled_pred, smooth), label='NuSVR Prediction')
plt.plot(movingaverage(real_value, smooth), label='NYSE Closing Price')

plt.legend()
plt.show()
