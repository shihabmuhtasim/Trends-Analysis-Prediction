
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

# Calculate the sizes for train, test, and validation sets
train_size = int(len(df_for_training_scaled) * 0.7)
test_size = int(len(df_for_training_scaled) * 0.2)
val_size = len(df_for_training_scaled) - train_size - test_size

# Split the data into train, test, and validation sets
train_data = df_for_training_scaled[:train_size]
test_data = df_for_training_scaled[train_size:train_size + test_size]
val_data = df_for_training_scaled[train_size + test_size:]

# Empty lists to be populated using formatted training, testing, and validation data
trainX, trainY = [], []
testX, testY = [], []
valX, valY = [], []

n_future = 1   # Number of instances we want to look into the future based on the past data.
n_past = 14    # Number of past instances we want to use to predict the future.

# Prepare training data
for i in range(n_past, len(train_data) - n_future + 1):
    trainX.append(train_data[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(train_data[i + n_future - 1:i + n_future, 5])

# Prepare testing data
for i in range(n_past, len(test_data) - n_future + 1):
    testX.append(test_data[i - n_past:i, 0:df_for_training.shape[1]])
    testY.append(test_data[i + n_future - 1:i + n_future, 5])

# Prepare validation data
for i in range(n_past, len(val_data) - n_future + 1):
    valX.append(val_data[i - n_past:i, 0:df_for_training.shape[1]])
    valY.append(val_data[i + n_future - 1:i + n_future, 5])

# Convert to numpy arrays
trainX, trainY = np.array(trainX), np.array(trainY)
testX, testY = np.array(testX), np.array(testY)
valX, valY = np.array(valX), np.array(valY)

# Print shapes to verify
print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))
print('testX shape == {}.'.format(testX.shape))
print('testY shape == {}.'.format(testY.shape))
print('valX shape == {}.'.format(valX.shape))
print('valY shape == {}.'.format(valY.shape))

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

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

# Fit the model with validation data explicitly
history = model.fit(trainX, trainY, epochs=100, batch_size=12, 
                    validation_data=(valX, valY), verbose=1, 
                    callbacks=[early_stopping])

# Plot training & validation loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

# Predict the test data
test_predictions = model.predict(testX)

# Calculate Mean Squared Error for the test predictions
test_mse = mean_squared_error(testY, test_predictions)
print(f'Test MSE: {test_mse}')

# Calculate RMSE for the test predictions
test_rmse = np.sqrt(test_mse)
print(f'Test RMSE: {test_rmse}')

# Make predictions on the validation set
val_predictions = model.predict(valX)

# Calculate Mean Squared Error for the validation predictions
val_mse = mean_squared_error(valY, val_predictions)
print(f'Validation MSE: {val_mse}')

# Calculate RMSE for the validation predictions
val_rmse = np.sqrt(val_mse)
print(f'Validation RMSE: {val_rmse}')

# Plot Actual vs. Predicted values for the test set
plt.figure(figsize=(10, 6))
plt.plot(testY.flatten(), label='Actual')
plt.plot(test_predictions.flatten(), label='Predicted', color='green')
plt.title('Actual vs. Predicted Values (Test Set)')
plt.xlabel('Time Steps')
plt.ylabel('Scaled Value')
plt.legend()
plt.show()

# Plot Actual vs. Predicted values for the validation set
plt.figure(figsize=(10, 6))
plt.plot(valY.flatten(), label='Actual')
plt.plot(val_predictions.flatten(), label='Predicted', color='green')
plt.title('Actual vs. Predicted Values (Validation Set)')
plt.xlabel('Time Steps')
plt.ylabel('Scaled Value')
plt.legend()
plt.show()