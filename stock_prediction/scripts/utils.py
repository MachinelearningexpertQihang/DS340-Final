import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import yaml
import matplotlib.pyplot as plt

def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_sequences(data, seq_length):
    """
    Create sequences for time series prediction
    
    Args:
        data (numpy.ndarray): Input data
        seq_length (int): Sequence length
        
    Returns:
        tuple: (X, y) where X is the input sequences and y is the target values
    """
    xs, ys = [], []
    
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length, 0]  # Assuming the first column is the target (close price)
        xs.append(x)
        ys.append(y)
        
    return np.array(xs), np.array(ys)

class StockDataset(Dataset):
    """
    Dataset for stock price prediction
    """
    def __init__(self, X, y):
        """
        Initialize dataset
        
        Args:
            X (numpy.ndarray): Input sequences
            y (numpy.ndarray): Target values
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_dataloaders(X_train, y_train, X_test, y_test, batch_size):
    """
    Prepare data loaders for training and testing
    
    Args:
        X_train (numpy.ndarray): Training input sequences
        y_train (numpy.ndarray): Training target values
        X_test (numpy.ndarray): Testing input sequences
        y_test (numpy.ndarray): Testing target values
        batch_size (int): Batch size
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics
    
    Args:
        y_true (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
        
    Returns:
        dict: Dictionary of metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def plot_predictions(y_true, y_pred, title='Stock Price Prediction'):
    """
    Plot true vs predicted values
    
    Args:
        y_true (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
        title (str): Plot title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{title.replace(" ", "_").lower()}.png')
    plt.close()

def save_model(model, path):
    """
    Save model to disk
    
    Args:
        model (torch.nn.Module): Model to save
        path (str): Path to save the model
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """
    Load model from disk
    
    Args:
        model (torch.nn.Module): Model to load weights into
        path (str): Path to the saved model
        
    Returns:
        torch.nn.Module: Loaded model
    """
    model.load_state_dict(torch.load(path))
    return model