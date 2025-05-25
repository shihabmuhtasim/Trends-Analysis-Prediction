# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 00:34:29 2024

@author: 21301648
"""

import numpy as np
import pandas as pd
# import torch
# from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


df_concatenated = pd.read_csv("D:/21301648/tweetconcat_intovec.csv")

def string_to_array(string_embedding):
    # Convert the string representation of a numpy array back to a numpy array
    return np.fromstring(string_embedding.strip('[]'), sep=' ', dtype=np.float32)

# Apply the conversion to your embeddings
df_concatenated['bert_embedding'] = df_concatenated['bert_embedding'].apply(string_to_array)

# Verify the conversion
print(df_concatenated['bert_embedding'].apply(type).unique())

# Define a function to scale embeddings and then concatenate scaled frequency
def scale_and_append_frequency(row, embedding_scaler, frequency_scaler):
    embedding = row['bert_embedding']
    # Scale the embedding
    scaled_embedding = embedding_scaler.transform([embedding])[0]
    frequency = row['Frequency']
    # Scale the frequency
    scaled_frequency = frequency_scaler.transform([[frequency]])[0][0]
    # Convert scaled frequency to an array and concatenate
    return np.append(scaled_embedding, scaled_frequency)

# Prepare to collect embeddings and frequencies for scaling
embedding_list = np.array(df_concatenated['bert_embedding'].tolist())
frequency_list = np.array(df_concatenated['Frequency'].tolist()).reshape(-1, 1)

# Fit the scalers on all embeddings and frequencies
embedding_scaler = StandardScaler()
frequency_scaler = StandardScaler()  # or MinMaxScaler(feature_range=(0, 1))

embedding_scaler.fit(embedding_list)
frequency_scaler.fit(frequency_list)

# Apply scaling and appending frequency
df_concatenated['bert_embedding_with_frequency'] = df_concatenated.apply(
    scale_and_append_frequency, axis=1, embedding_scaler=embedding_scaler, frequency_scaler=frequency_scaler
)

# Verify maximum hour for sanity check
max_hour = df_concatenated['Hour'].max()
print("Maximum hour:", max_hour)

# Create a mapping for User Ids
user_id_mapping = {user_id: idx for idx, user_id in enumerate(df_concatenated['User Id'].unique())}
num_users = len(user_id_mapping)
num_hours = 39  # Assuming you have 39 hours
embedding_size = len(df_concatenated['bert_embedding_with_frequency'].iloc[0])

# Ensure correct data types
df_concatenated['User Id'] = df_concatenated['User Id'].astype(str)
df_concatenated['Hour'] = df_concatenated['Hour'].astype(int)

# Initialize 3D numpy array
lstm_input = np.zeros((num_users, num_hours, embedding_size))

# Populate the 3D array
for _, row in df_concatenated.iterrows():
    user_id = row['User Id']
    hour = row['Hour'] - 1  # Convert to zero-based index
    embedding = row['bert_embedding_with_frequency']
    
    # Get the user index
    user_index = user_id_mapping[user_id]
    
    # Assign embedding to the 3D array
    lstm_input[user_index, hour, :] = embedding

# Print the shape of the resulting array
print(lstm_input.shape)

# Print the data for the first user
first_user_index = 0  # Adjust if needed
print(f"Data for the first user (index {first_user_index}):")
print(lstm_input[first_user_index])




# Calculate the split indices
train_size = int(0.7 * len(lstm_input))  # 70% for training
val_size = int(0.1 * len(lstm_input))    # 10% for validation
test_size = len(lstm_input) - train_size - val_size  # Remaining 20% for testing

# Split the data
train_data = lstm_input[:train_size]  # First 70% for training
val_data = lstm_input[train_size:train_size + val_size]  # Next 10% for validation
test_data = lstm_input[train_size + val_size:]  # Remaining 20% for testing

# Print the shapes of the split data
print("Train data shape:", train_data.shape)     # Should print: (67156, 39, 769) assuming 70% of 95938
print("Validation data shape:", val_data.shape)  # Should print: (9593, 39, 769) assuming 10% of 95938
print("Test data shape:", test_data.shape)       # Should print: (19189, 39, 769) remaining 20%


# Create train_x and train_y
train_x = train_data[:, :38, :]
train_y = train_data[:, 38:, :]

test_x = test_data[:, :38, :]
test_y = test_data[:, 38:, :]

val_x = val_data[:, :38, :]
val_y = val_data[:, 38:, :]

# Print the shapes of train_x and train_y
print("Train x shape:", train_x.shape)
print("Train y shape:", train_y.shape)
print("Test x shape:", test_x.shape)
print("Test y shape:", test_y.shape)
print("Validation x shape:", val_x.shape)
print("Validation y shape:", val_y.shape)


# # FOR TESTING WITHOUT FREQUENCY
# # Create train_x, val_x, and test_x excluding the last column
# train_x = train_data[:, :38, :-1]  # Exclude the last column for the first 38 hours
# val_x = val_data[:, :38, :-1]      # Exclude the last column for the first 38 hours
# test_x = test_data[:, :38, :-1]    # Exclude the last column for the first 38 hours



train_y = train_y[:, :, -1].reshape(train_y.shape[0], 1, 1)
test_y = test_y[:, :, -1].reshape(test_y.shape[0], 1, 1)
val_y = val_y[:, :, -1].reshape(val_y.shape[0], 1, 1)

# Print the shapes of train_x and train_y
print("Train x shape:", train_x.shape)
print("Train y shape:", train_y.shape)
print("Test x shape:", test_x.shape)
print("Test y shape:", test_y.shape)
print("Validation x shape:", val_x.shape)
print("Validation y shape:", val_y.shape)




# import tensorflow as tf

# # Clear the previous session
# from tensorflow.keras import backend as K
# K.clear_session()

# # Set memory growth for GPU (if using GPU)
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError as e:
#         print(f"Error: {e}")

# # Enable mixed precision
# from tensorflow.keras.mixed_precision import Policy, set_global_policy

# # Set the global mixed precision policy
# policy = Policy('mixed_float16')
# set_global_policy(policy)

# print(f"Mixed precision policy set to {policy.name}")

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense


# Create a Sequential model
model = Sequential()

# Add a Bidirectional LSTM layer
# You can set `return_sequences=True` if you need to stack more LSTM layers or if you want to return the full sequence
model.add(Bidirectional(LSTM(39, return_sequences=False), input_shape=(train_x.shape[1], train_x.shape[2])))

# Add a Dense layer (fully connected layer) after the LSTM layers
model.add(Dense(20, activation='relu'))

# Output layer
model.add(Dense(1))  # Assuming you're doing regression, change the number of units if necessary

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Summary of the model
model.summary()

from tensorflow.keras.callbacks import EarlyStopping

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(8)
# val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(8)

# history = model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[early_stopping])

# Fit the model with early stopping
history = model.fit(train_x, train_y, epochs=100, batch_size=32, validation_data=(val_x, val_y), callbacks=[early_stopping])

# Plot train and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()


# Predict on the test data
predictions = model.predict(test_x)

# Evaluate the model
mse = model.evaluate(test_x, test_y)
print('Mean Squared Error:', mse)

predictions = predictions.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)


# Invert the scaling to get the actual values
predictions_actual = frequency_scaler.inverse_transform(np.concatenate((test_x[:, -1, :-1], predictions), axis=1))[:, -1]
test_y_actual = frequency_scaler.inverse_transform(np.concatenate((test_x[:, -1, :-1], test_y), axis=1))[:, -1]


plt.plot(test_y_actual, label='Actual')
plt.plot(predictions_actual, label='Predicted')
plt.legend()
plt.show()


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Calculate MSE
mse = mean_squared_error(test_y_actual, predictions_actual)

# Calculate RMSE
rmse = mean_squared_error(test_y_actual, predictions_actual, squared=False)

# Calculate MAE
mae = mean_absolute_error(test_y_actual, predictions_actual)

# Calculate R-squared
r2 = r2_score(test_y_actual, predictions_actual)

# Print the results
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R-squared: {r2}")




