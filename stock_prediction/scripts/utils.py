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
    Calculate evaluation metrics for volatility prediction
    
    Args:
        y_true (numpy.ndarray): True volatility values
        y_pred (numpy.ndarray): Predicted volatility values
        
    Returns:
        dict: Dictionary of metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Basic metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Volatility-specific metrics
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
    
    # Directional Accuracy (for volatility trend)
    vol_direction_true = np.diff(y_true) > 0
    vol_direction_pred = np.diff(y_pred) > 0
    directional_accuracy = np.mean(vol_direction_true == vol_direction_pred) * 100
    
    # QLIKE loss (commonly used in volatility forecasting)
    qlike = np.mean(y_true/y_pred - np.log(y_true/y_pred) - 1)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE (%)': mape,
        'Directional Accuracy (%)': directional_accuracy,
        'QLIKE': qlike
    }

def plot_predictions(y_true, y_pred, title='Volatility Prediction'):
    """
    Plot true vs predicted volatility values
    
    Args:
        y_true (numpy.ndarray): True volatility values
        y_pred (numpy.ndarray): Predicted volatility values
        title (str): Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Actual vs Predicted Volatility
    ax1.plot(y_true, label='Actual Volatility', color='blue', alpha=0.7)
    ax1.plot(y_pred, label='Predicted Volatility', color='red', alpha=0.7)
    ax1.set_title('Actual vs Predicted Volatility')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Volatility (%)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Prediction Error
    error = y_true - y_pred
    ax2.plot(error, label='Prediction Error', color='green', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_title('Prediction Error')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Error')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{title.replace(" ", "_").lower()}.png')
    plt.close()

def plot_backtest_results(backtest_df):
    """
    Plot backtest results for volatility trading
    
    Args:
        backtest_df (pd.DataFrame): Dataframe with backtest results
    """
    # Create figure with multiple subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot 1: Strategy Cumulative Returns
    cum_returns = (1 + backtest_df['Strategy Returns']).cumprod()
    cum_market = (1 + backtest_df['Market Returns']).cumprod()
    
    ax1.plot(cum_returns, label='Strategy Returns', color='blue')
    ax1.plot(cum_market, label='Market Returns', color='gray', alpha=0.7)
    ax1.set_title('Cumulative Returns')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Predicted Volatility and Position Sizes
    ax2.plot(backtest_df['Predicted Volatility'], label='Predicted Volatility', color='red')
    ax2.set_title('Predicted Volatility')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Volatility')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(backtest_df['Position'], label='Position Size', color='green', alpha=0.5)
    ax2_twin.set_ylabel('Position Size')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2)
    ax2.grid(True)
    
    # Plot 3: Drawdown Analysis
    cum_returns = (1 + backtest_df['Strategy Returns']).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    
    ax3.fill_between(range(len(drawdowns)), drawdowns, 0, color='red', alpha=0.3)
    ax3.set_title('Strategy Drawdown')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Drawdown')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/backtest_analysis.png')
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
    model.load_state_dict(torch.load(path, weights_only=False))
    return model