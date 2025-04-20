"""
Utility functions for stock price prediction model
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping

def preprocess_data(data_path):
    """
    Load and preprocess stock data
    """
    data = pd.read_csv(data_path)
    new_data = data.iloc[:, [0, 5]].copy()
    
    # Convert 'Close' column to float to avoid categorical warnings
    new_data['Close'] = pd.to_numeric(new_data['Close'], errors='coerce')
    
    # setting index
    new_data.index = pd.to_datetime(new_data.Date)
    new_data.drop('Date', axis=1, inplace=True)
    
    return new_data

def create_sequences(data, sequence_length, split_point):
    """
    Create sequence data for LSTM model
    """
    dataset = data.values
    train = dataset[0:split_point, :]
    test = dataset[split_point:, :]
    
    # Apply feature scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Create training sequences
    x_train, y_train = [], []
    for i in range(sequence_length, len(train)):
        x_train.append(scaled_data[i-sequence_length:i, 0])
        y_train.append(scaled_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, train, test, scaler, dataset

def build_lstm_model(input_shape):
    """
    Build and compile LSTM model
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        LSTM(units=50),
        Dense(1)
    ])
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def prepare_test_data(data, test_length, sequence_length, scaler):
    """
    Prepare test data for prediction
    """
    inputs = data[len(data) - test_length - sequence_length:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    
    X_test = []
    for i in range(sequence_length, inputs.shape[0]):
        X_test.append(inputs[i-sequence_length:i, 0])
    
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    return X_test

def evaluate_model(actual, predicted):
    """
    Calculate model performance metrics
    """
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    accuracy = r2 * 100
    
    return {
        'rmse': rmse,
        'r2': r2,
        'accuracy': accuracy
    }

def plot_results(train_data, test_data, predictions, output_path):
    """
    Plot and save the prediction results
    """
    plt.figure(figsize=(14, 8))
    plt.plot(train_data.index, train_data['Close'], 'b-', label='Training Data')
    plt.plot(test_data.index, test_data['Close'], 'g-', label='Actual Price')
    plt.plot(test_data.index, predictions, 'r--', label='Predicted Price')
    plt.title('Stock Price Prediction', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (INR)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

def plot_training_loss(history, output_path):
    """
    Plot training loss history
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.title('Model Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.savefig(output_path)

def save_prediction_results(test_data, predictions, output_path):
    """
    Save prediction results to CSV
    """
    # Ensure predictions is 1-dimensional
    if hasattr(predictions, 'shape') and len(predictions.shape) > 1:
        predictions = predictions.flatten()
        
    results_df = pd.DataFrame({
        'Date': test_data.index,
        'Actual': test_data['Close'],
        'Predicted': predictions
    })
    results_df.to_csv(output_path, index=False) 