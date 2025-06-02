import math
import tensorflow as tf
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import RMSprop
from keras.initializers import glorot_uniform, glorot_normal, RandomUniform
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from pandas.core.indexes import interval
yf.pdr_override()


ticker = ["AMZN"]
years = 10*365
columns = ['Close', 'High', 'Low', 'Open']
testCases = 10
startDate = dt.datetime.today() - dt.timedelta(days=years)
endDate = dt.datetime.today()
sc = MinMaxScaler()
X_train = []
Y_train = []
X_test = []
Y_test = []
dates = []

regressor = Sequential()


def GetData(ticker, years):
    data = pd.DataFrame(pdr.data.get_data_yahoo(
        ticker, startDate.date(), endDate.date(), interval='1wk'))
    AVG = (data['High'] + data['Low']) / 2
    data['Average'] = AVG
    data.index = data.index.date
    return data


data = GetData(ticker, years)


def RemoveColumns(data, columns):
    attributes = list(data)[0:7]
    for col in columns:
        attributes.remove(col)
    data = data[attributes].astype(float)
    return data


data = RemoveColumns(data, columns)

print(data.head(7))
print(data.shape)


def ScaleData(data):
    scaled_data = sc.fit_transform(data)
    return scaled_data


scaled_data = ScaleData(data)


def TrainModel(testCases, X_train, Y_train):
    for i in range(1, data.shape[0]-testCases):
        X_train.append(scaled_data[i-1: i, 0: scaled_data.shape[1]])
        Y_train.append(scaled_data[i: i + 1, -1])
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    return X_train, Y_train


def TestModel(testCases, X_test, Y_test, dates):
    for i in range(data.shape[0]-testCases, data.shape[0]):
        X_test.append(scaled_data[i-1: i, 0: scaled_data.shape[1]])
        Y_test.append(scaled_data[i: i + 1, -1])
        dates.append(data.index[i: i + 1])
    X_test, Y_test, dates = np.array(X_test), np.array(Y_test), np.array(dates)
    return X_test, Y_test, dates


X_train, Y_train = TrainModel(testCases, X_train, Y_train)
X_test, Y_test, dates = TestModel(testCases, X_test, Y_test, dates)


def RunModel(X_train, Y_train):
    # optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    optimizer = 'adam'
    init = glorot_normal(seed=None)
    init1 = RandomUniform(minval=-0.05, maxval=0.05)

    regressor.add(LSTM(units=64, return_sequences=True, input_shape=(
        X_train.shape[1], X_train.shape[2]), kernel_initializer=init))
    regressor.add(Dropout(0.4))

    regressor.add(LSTM(units=64, return_sequences=True,
                  kernel_initializer=init))
    regressor.add(Dropout(0.3))

    regressor.add(LSTM(units=100, return_sequences=False,
                  kernel_initializer=init))
    regressor.add(Dropout(0.3))

    regressor.add(
        Dense(Y_train.shape[1], activation='linear', kernel_initializer=init1))

    regressor.compile(optimizer=optimizer, loss='mean_squared_error')

    regressor.fit(X_train, Y_train, epochs=90, batch_size=16)


RunModel(X_train, Y_train)

predictedStockPrices = regressor.predict(X_test)


expected = np.asarray(data['Average'])
expected = expected[data.shape[0]-testCases: data.shape[0]]
expected = np.reshape(expected, (expected.shape[0], 1))

predictedStockPrices_copy = np.repeat(
    predictedStockPrices, scaled_data.shape[1], axis=-1)
predictions = sc.inverse_transform(predictedStockPrices_copy)[:, 0]
predictions = np.reshape(predictions, (predictions.shape[0], 1))


plt.plot(expected, color='blue', label='expected')
plt.plot(predictions, color='red', label='predicted')
plt.legend()
plt.show()


print("\n(expected) to (predicted) \n", pd.DataFrame(
    predictions[:, 0], expected[:, 0]))


data1 = GetData(ticker, years)

print(data1[- testCases:])

gap = []

low = np.asarray(data1['Low'])
low = low[data1.shape[0]-testCases: data1.shape[0]]

high = np.asarray(data1['High'])
high = high[data1.shape[0]-testCases: data1.shape[0]]


def measure_within_range(pred_values, high_values, low_values):
    """
    Calculates the percentage of predicted values that are within their corresponding high and low values.
    """
    # Count the number of predicted values within range
    num_within_range = 0
    for pred, high, low in zip(pred_values, high_values, low_values):
        if low <= pred <= high:
            num_within_range += 1

    # Calculate the percentage within range
    percentage_within_range = (num_within_range / len(pred_values)) * 100

    return percentage_within_range


percentage_within_range = measure_within_range(
    predictions, high, low)

print("\n"f"{percentage_within_range}% of predicted values are within their corresponding high and low values. \n")

# print("R2: %r " % round((r2_score(expected, predictions)*100), 4), "\n")
# gap = high - low

# p = predictions.flatten()

# diff_from_low = []
# diff_from_low = np.subtract(p, low)
# print("\n(diff_from_low) to (gap) \n",
#       pd.DataFrame(gap[:], diff_from_low[:]), "\n")
# gap = np.round(gap)
# diff_from_low = np.round(diff_from_low)

# rights = 0
# total = 0
# for x in range(0, diff_from_low.shape[0]):
#     if (diff_from_low[x] < 0 or diff_from_low[x] > gap[x]):
#         print("testcase number", x+1, ": ", "False")
#     else:
#         print("testcase number", x+1, ": ", "True")
#         rights += 1
#     total += 1
# print("\nPrecentage of rights: ", (rights/(x+1))*100, "%")

# def count_within_threshold(predicted, actual, threshold):
#     diff_high = (predicted - actual['High']).abs() / actual['High']
#     diff_low = (predicted - actual['Low']).abs() / actual['Low']
#     within_threshold = (diff_high < threshold) & (diff_low < threshold)
#     return within_threshold.sum()


# threshold = 0.05
# within_threshold = count_within_threshold(
#     predictions, data.iloc[-testCases:], threshold)

# print(f"{within_threshold}/{testCases} ({within_threshold/testCases:.2%}) predicted values are within {threshold:.2%} of their actual high and low values")
