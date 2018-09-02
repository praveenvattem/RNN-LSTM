# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 22:42:21 2018

@author: GANESHA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the training set
dataset_train=pd.read_csv('stock.csv')
training_set1 =dataset_train.iloc[:,2:3].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled1 = sc.fit_transform(training_set1)



X_train1 = []
y_train1 = []
y_train=[]
for i in range(60, 999):
    X_train1.append(training_set_scaled1[i-60:i, 0])
    y_train1.append(training_set_scaled1[i, 0])
X_train1, y_train1 = np.array(X_train1), np.array(y_train1)

X_train1 = np.reshape(X_train1, (X_train1.shape[0], X_train1.shape[1], 1))




from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train1.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train1, y_train, epochs = 10, batch_size = 20)

dataset_test = pd.read_csv('stock_test.csv')
dataset_test1 = pd.read_csv('stock_test.csv')
real_stock_price_open = dataset_test1.iloc[:, 2:3].values
dataset_total = pd.concat((dataset_train['OPEN'], dataset_test['OPEN']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test1 = []
for i in range(60, 81):
    X_test1.append(inputs[i-60:i, 0])
X_test1 = np.array(X_test1)
X_test1 = np.reshape(X_test1, (X_test1.shape[0], X_test1.shape[1], 1))
predicted_stock_price_open = regressor.predict(X_test1)
predicted_stock_price_open = sc.inverse_transform(predicted_stock_price_open)


plt.plot(real_stock_price_open, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price_open, color = 'blue', label = 'Predicted  Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()