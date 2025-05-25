# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:40:45 2024

@author: 21301648
"""

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU is available. Using {torch.cuda.get_device_name(device)}")
else:
    device = torch.device('cpu')
    print("GPU is not available. Using CPU instead.")

# Example: Load your DataFrame

df = pd.read_csv("D:/21301648/data_lstm_processed_all_perhour.csv")

# Convert 'Tweet Posted Time' to datetime
df['Tweet Posted Time'] = pd.to_datetime(df['Tweet Posted Time'], errors='coerce')


# Drop the column 'Unnamed: 0'
df.drop(columns=['Unnamed: 0'], inplace=True)

df.rename(columns={'User  Id': 'User Id'}, inplace=True)

# List of columns to keep
columns_to_keep = ["Tweet Posted Time", "Hour", "User Id", "Tweet Content", "Frequency"]

# Create a new DataFrame with only the specified columns
df = df[columns_to_keep]

# Group by 'User Id' and 'Hour', concatenate the 'Tweet Content', and keep the maximum 'Frequency'
df_concatenated = (
    df.groupby(['User Id', 'Hour'])
    .agg({
        'Tweet Content': ' '.join,
        'Frequency': 'max'  # Assuming all 'Frequency' values are the same for tweets in the same hour
    })
    .reset_index()
)

# Rename the concatenated column if desired
df_concatenated.rename(columns={'Tweet Content': 'Concatenated Tweet Content'}, inplace=True)



# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Function to encode tweets
def encode_tweets(tweets):
    encoded_inputs = tokenizer(
        tweets,  # List of tweet contents
        padding=True,  # Pad to the longest sequence
        truncation=True,  # Truncate to the model's max length
        max_length=512,  # Max length for BERT
        return_tensors='pt'  # Return PyTorch tensors
    )
    return encoded_inputs

# Function to get BERT embeddings
def get_bert_embeddings(encoded_inputs):
    input_ids = encoded_inputs['input_ids'].to(device)
    attention_mask = encoded_inputs['attention_mask'].to(device)
    
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # Get the embeddings from the last hidden state (use [CLS] token)
    last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)
    cls_embeddings = last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
    
    return cls_embeddings

# Define batch size
batch_size = 32  # Adjust this size based on your GPU memory availability

# Initialize an empty list to store BERT embeddings
bert_embeddings_list = []

# Process tweets in batches
for i in range(0, len(df_concatenated), batch_size):
    batch_tweets = df_concatenated['Concatenated Tweet Content'][i:i+batch_size].tolist()
    encoded_batch = encode_tweets(batch_tweets)
    batch_embeddings = get_bert_embeddings(encoded_batch)
    bert_embeddings_list.extend(batch_embeddings.cpu().numpy())  # Move to CPU and store
    
# Add the BERT embeddings to the DataFrame
df_concatenated['bert_embedding'] = bert_embeddings_list

# Display the DataFrame with embeddings
print(df_concatenated.head())

# Print the shape of the first BERT embedding
first_embedding = df_concatenated['bert_embedding'].iloc[0]
print(f"Shape of the first BERT embedding: {first_embedding.shape}")


# Save the updated DataFrame if needed
# df_concatenated.to_csv("D:/21301648/tweetconcat_intovec.csv", index=False)


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
# first_user_index = 0  # Adjust if needed
# print(f"Data for the first user (index {first_user_index}):")
# print(lstm_input[first_user_index])




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



# Labels: frequency of the 39th hour (last feature)
train_x = train_data[:, :38, :]  # All features for the first 38 hours
train_y = train_data[:, 38, -1]  # Frequency of the 39th hour (last feature)

val_x = val_data[:, :38, :]  # All features for the first 38 hours
val_y = val_data[:, 38, -1]  # Frequency of the 39th hour (last feature)

test_x = test_data[:, :38, :]  # All features for the first 38 hours
test_y = test_data[:, 38, -1]  # Frequency of the 39th hour (last feature)

# Ensure labels are in the correct shape (N, 1)
train_y = train_y.reshape(-1, 1)
val_y = val_y.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)

# Print the shapes of train_x, train_y, test_x, test_y, val_x, val_y
print("Train x shape:", train_x.shape)  # (67156, 38, 769)
print("Train y shape:", train_y.shape)  # (67156, 1)
print("Test x shape:", test_x.shape)    # (19189, 38, 769)
print("Test y shape:", test_y.shape)    # (19189, 1)
print("Validation x shape:", val_x.shape)  # (9593, 38, 769)
print("Validation y shape:", val_y.shape)  # (9593, 1)






import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt



# Assuming train_x, train_y, val_x, val_y, test_x, and test_y are already prepared
train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32).view(-1, 1)
val_x = torch.tensor(val_x, dtype=torch.float32)
val_y = torch.tensor(val_y, dtype=torch.float32).view(-1, 1)
test_x = torch.tensor(test_x, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32).view(-1, 1)

# Define the Bi-directional LSTM model with Dropout and bottlenecking
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_layers, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size1,
                             num_layers=num_layers,
                             dropout=dropout,
                             bidirectional=True,
                             batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size1 * 2,  # Bidirectional means input size is doubled
                             hidden_size=hidden_size2,
                             num_layers=num_layers,
                             dropout=dropout,
                             bidirectional=True,
                             batch_first=True)
        self.fc = nn.Linear(hidden_size2 * 2, 1)  # *2 for bidirectional

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.fc(out[:, -1, :])  # Get the output from the last time step
        return out

# Initialize the model
input_size = train_x.shape[2]  # 769 features
hidden_size1 = 64  # Number of units in the first LSTM layer
hidden_size2 = 32   # Number of units in the second LSTM layer
num_layers = 2      # Number of LSTM layers
dropout = 0.5
model = BiLSTM(input_size, hidden_size1, hidden_size2, num_layers, dropout)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create datasets and dataloaders
batch_size = 64
train_dataset = TensorDataset(train_x, train_y)
val_dataset = TensorDataset(val_x, val_y)
test_dataset = TensorDataset(test_x, test_y)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training and validation
num_epochs = 20
early_stopping_patience = 5
best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
    
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

# Plot training and validation loss
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()



# Load the best model for testing
model.load_state_dict(torch.load('best_model.pth'))

# Define the loss criterion (same as used during training)
criterion = torch.nn.MSELoss()  # or the loss function you used during training

# Prepare to collect predictions and actual values
predictions = []
actuals = []

# Evaluate on the test set and calculate test loss
model.eval()
test_loss = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item() * inputs.size(0)
        
        # Collect predictions and actual values
        predictions.append(outputs.numpy())
        actuals.append(targets.numpy())

# Average the test loss
test_loss /= len(test_loader.dataset)
print(f'Test Loss: {test_loss:.4f}')

# Convert lists to numpy arrays
predictions = np.concatenate(predictions, axis=0)
actuals = np.concatenate(actuals, axis=0)

# Flatten the arrays if needed for plotting
predictions = predictions.flatten()
actuals = actuals.flatten()

# If you applied scaling during preprocessing, inverse scale the predictions and actuals
# Assume you have scalers for frequency
predictions = frequency_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
actuals = frequency_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

# Calculate evaluation metrics
mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(actuals, predictions)

# Print the results
print(f'MSE: {mse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R²: {r2:.4f}')

# Plot the actual vs. predicted data
plt.figure(figsize=(12, 6))
plt.plot(actuals, label='Actual Data', color='blue')
plt.plot(predictions, label='Predicted Data', color='red', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Frequency')
plt.title('Actual vs. Predicted Frequency for the 39th Hour')
plt.legend()
plt.show()






