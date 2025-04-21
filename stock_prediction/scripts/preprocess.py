import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from scripts.utils import create_sequences, load_config

def preprocess_data(config_path='config.yaml'):
    """
    Preprocess stock data for training and testing
    
    Args:
        config_path (str): Path to config file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create directories if they don't exist
    os.makedirs(config['paths']['processed_data_dir'], exist_ok=True)
    
    # Load raw data
    data_path = os.path.join(config['paths']['raw_data_dir'], config['data']['filename'])
    df = pd.read_csv(data_path)
    
    # Skip the second row which contains duplicate headers
    if df.iloc[0].equals(pd.Series(['AAPL'] * len(df.columns))):
        df = df.iloc[1:].reset_index(drop=True)
    
    # Convert Datetime column
    df['Date'] = pd.to_datetime(df['Datetime'])
    df.sort_values('Date', inplace=True)
    
    # Select features and ensure correct column names
    features = config['data']['features']
    df = df.rename(columns={'Datetime': 'Date'})
    
    # Convert all feature columns to numeric, removing any non-numeric characters
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df[['Date'] + features]
    
    # Fill missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # Add technical indicators if specified
    if config['data']['add_technical_indicators']:
        # Add Moving Averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # Add Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Add MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Add Bollinger Bands
        df['20d_std'] = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['MA20'] + (df['20d_std'] * 2)
        df['Lower_Band'] = df['MA20'] - (df['20d_std'] * 2)
    
    # Drop rows with NaN values (due to rolling windows)
    df.dropna(inplace=True)
    
    # Drop Date column before scaling
    df = df.drop('Date', axis=1)
    
    # Split data into train and test sets
    train_size = int(len(df) * config['data']['train_test_split'])
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    
    # Normalize data using MinMaxScaler (fit on train_data only)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data.values)
    test_scaled = scaler.transform(test_data.values)
    
    # Create sequences
    X_train, y_train = create_sequences(train_scaled, config['data']['seq_length'])
    X_test, y_test = create_sequences(test_scaled, config['data']['seq_length'])
    
    # Save processed data
    train_data_path = os.path.join(config['paths']['processed_data_dir'], 'train_data.pt')
    test_data_path = os.path.join(config['paths']['processed_data_dir'], 'test_data.pt')
    
    torch.save({
        'X': torch.tensor(X_train, dtype=torch.float32),
        'y': torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1),
        'scaler': scaler  # Save the scaler for inverse transform
    }, train_data_path)
    
    torch.save({
        'X': torch.tensor(X_test, dtype=torch.float32),
        'y': torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1),
        'scaler': scaler  # Save scaler for later use
    }, test_data_path)
    
    print(f"Processed data saved to {train_data_path} and {test_data_path}")
    print(f"Train data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    
    # Save feature names for reference
    with open(os.path.join(config['paths']['processed_data_dir'], 'features.txt'), 'w') as f:
        f.write('\n'.join(df.columns))
    
    return train_data_path, test_data_path

if __name__ == "__main__":
    preprocess_data()
