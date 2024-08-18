# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns


df = pd.read_csv('C:/Users/21301610/Downloads/21301610/data_lstm_processed.csv')

#df = df.sample(frac=0.2, random_state=42) 

cols = list(df)[1:8]

#New dataframe with only training data - 7 columns
df_for_training = df[cols].astype(float)

df_for_training.fillna(df_for_training.mean(), inplace=True)

# normalize the dataset
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

# Split into training and testing sets (80% train, 20% test)
train_size = int(len(df_for_training_scaled) * 0.8)
train_data, test_data = df_for_training_scaled[:train_size], df_for_training_scaled[train_size:]

# Empty lists to be populated using formatted training data
trainX, trainY = [], []
testX, testY = [], []

n_future = 1   # Number of days we want to look into the future based on the past days.
n_past = 14    # Number of past days we want to use to predict the future.

# Prepare training data
for i in range(n_past, len(train_data) - n_future + 1):
    trainX.append(train_data[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(train_data[i + n_future - 1:i + n_future, 5])

# Prepare testing data
for i in range(n_past, len(test_data) - n_future + 1):
    testX.append(test_data[i - n_past:i, 0:df_for_training.shape[1]])
    testY.append(test_data[i + n_future - 1:i + n_future, 5])

trainX, trainY = np.array(trainX), np.array(trainY)
testX, testY = np.array(testX), np.array(testY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))
print('testX shape == {}.'.format(testX.shape))
print('testY shape == {}.'.format(testY.shape))


import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

# Compile the model with a lower learning rate for stability
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

model.summary()

# Implement Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model with Early Stopping and a reasonable initial number of epochs
history = model.fit(trainX, trainY, epochs=100, batch_size=12, validation_split=0.1, verbose=1, callbacks=[early_stopping])

# Plot training & validation loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()


# Predict the test data
test_predictions = model.predict(testX)

from sklearn.metrics import mean_squared_error

# Calculate Mean Squared Error for the test predictions
mse = mean_squared_error(testY, test_predictions)
print(f'Test MSE: {mse}')

# If needed, you can also convert the MSE to RMSE (Root Mean Squared Error)
rmse = np.sqrt(mse)
print(f'Test RMSE: {rmse}')

# Plot Actual vs. Predicted values
plt.figure(figsize=(10, 6))
plt.plot(testY.flatten(), label='Actual')
plt.plot(test_predictions.flatten(), label='Predicted', color='green')
plt.title('Actual vs. Predicted Values')
plt.xlabel('Time Steps')
plt.ylabel('Scaled Value')
plt.legend()
plt.show()
