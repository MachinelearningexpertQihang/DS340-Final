import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models.gru import GRUModel
from models.transformer import TransformerModel
from models.gru_transformer import GRUTransformerModel
from scripts.utils import load_config, calculate_metrics, plot_predictions, load_model

def evaluate_model(config_path='config.yaml', model_path=None):
    """
    Evaluate the model on test data
    
    Args:
        config_path (str): Path to config file
        model_path (str, optional): Path to model checkpoint. If None, use the best model.
        
    Returns:
        dict: Evaluation metrics
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config['training']['use_gpu'] else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    test_data_path = os.path.join(config['paths']['processed_data_dir'], 'test_data.pt')
    test_data = torch.load(test_data_path, weights_only=False)
    X_test, y_test = test_data['X'], test_data['y']
    scaler = test_data['scaler']
    
    # Move data to device
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Get input dimensions
    input_dim = X_test.shape[2]  # Number of features
    output_dim = y_test.shape[1]  # Number of output dimensions
    
    # Initialize model based on configuration
    model_type = config['model']['type']
    
    if model_type == 'gru':
        model = GRUModel(
            input_dim=input_dim,
            hidden_dim=config['model']['gru']['hidden_dim'],
            num_layers=config['model']['gru']['num_layers'],
            output_dim=output_dim,
            dropout=config['model']['dropout']
        )
    elif model_type == 'transformer':
        model = TransformerModel(
            input_dim=input_dim,
            d_model=config['model']['transformer']['d_model'],
            nhead=config['model']['transformer']['nhead'],
            num_layers=config['model']['transformer']['num_layers'],
            dim_feedforward=config['model']['transformer']['dim_feedforward'],
            output_dim=output_dim,
            dropout=config['model']['dropout'],
            max_len=config['data']['seq_length']
        )
    elif model_type == 'gru_transformer':
        model = GRUTransformerModel(
            input_dim=input_dim,
            gru_hidden_dim=config['model']['gru']['hidden_dim'],
            gru_num_layers=config['model']['gru']['num_layers'],
            transformer_d_model=config['model']['transformer']['d_model'],
            transformer_nhead=config['model']['transformer']['nhead'],
            transformer_num_layers=config['model']['transformer']['num_layers'],
            transformer_dim_feedforward=config['model']['transformer']['dim_feedforward'],
            output_dim=output_dim,
            dropout=config['model']['dropout'],
            max_len=config['data']['seq_length']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model weights
    if model_path is None:
        model_path = os.path.join(config['paths']['model_dir'], 'best_model.pth')
    
    model = load_model(model, model_path)
    model = model.to(device)
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        y_pred = model(X_test)
    
    # Move tensors to CPU for evaluation
    y_test_cpu = y_test.cpu().numpy()
    y_pred_cpu = y_pred.cpu().numpy()
    
    # Inverse transform predictions if needed
    if config['evaluation']['inverse_transform']:
        # Create dummy arrays with the same shape as the original data
        y_test_dummy = np.zeros((len(y_test_cpu), scaler.n_features_in_))
        y_pred_dummy = np.zeros((len(y_pred_cpu), scaler.n_features_in_))
        
        # Put the predicted values in the first column (assuming Close price is the target)
        y_test_dummy[:, 0] = y_test_cpu.flatten()
        y_pred_dummy[:, 0] = y_pred_cpu.flatten()
        
        # Inverse transform
        y_test_inv = scaler.inverse_transform(y_test_dummy)[:, 0]
        y_pred_inv = scaler.inverse_transform(y_pred_dummy)[:, 0]
    else:
        y_test_inv = y_test_cpu.flatten()
        y_pred_inv = y_pred_cpu.flatten()
    
    # Calculate metrics
    metrics = calculate_metrics(y_test_inv, y_pred_inv)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # Plot predictions
    plot_predictions(y_test_inv, y_pred_inv, title=f'Stock Price Prediction ({model_type})')
    
    # Save predictions to CSV
    os.makedirs('results', exist_ok=True)
    pd.DataFrame({
        'Actual': y_test_inv,
        'Predicted': y_pred_inv
    }).to_csv('results/predictions.csv', index=False)
    
    return metrics

def predict_future(config_path='config.yaml', model_path=None, days_ahead=30):
    """
    Make future predictions
    
    Args:
        config_path (str): Path to config file
        model_path (str, optional): Path to model checkpoint. If None, use the best model.
        days_ahead (int): Number of days to predict ahead
        
    Returns:
        numpy.ndarray: Future predictions
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config['training']['use_gpu'] else 'cpu')
    
    # Load test data
    test_data_path = os.path.join(config['paths']['processed_data_dir'], 'test_data.pt')
    test_data = torch.load(test_data_path, weights_only=False)
    X_test, y_test = test_data['X'], test_data['y']
    scaler = test_data['scaler']
    
    # Get the last sequence from test data
    last_sequence = X_test[-1:].to(device)
    
    # Get input dimensions
    input_dim = X_test.shape[2]  # Number of features
    output_dim = y_test.shape[1]  # Number of output dimensions
    
    # Initialize model based on configuration
    model_type = config['model']['type']
    
    if model_type == 'gru':
        model = GRUModel(
            input_dim=input_dim,
            hidden_dim=config['model']['gru']['hidden_dim'],
            num_layers=config['model']['gru']['num_layers'],
            output_dim=output_dim,
            dropout=config['model']['dropout']
        )
    elif model_type == 'transformer':
        model = TransformerModel(
            input_dim=input_dim,
            d_model=config['model']['transformer']['d_model'],
            nhead=config['model']['transformer']['nhead'],
            num_layers=config['model']['transformer']['num_layers'],
            dim_feedforward=config['model']['transformer']['dim_feedforward'],
            output_dim=output_dim,
            dropout=config['model']['dropout'],
            max_len=config['data']['seq_length']
        )
    elif model_type == 'gru_transformer':
        model = GRUTransformerModel(
            input_dim=input_dim,
            gru_hidden_dim=config['model']['gru']['hidden_dim'],
            gru_num_layers=config['model']['gru']['num_layers'],
            transformer_d_model=config['model']['transformer']['d_model'],
            transformer_nhead=config['model']['transformer']['nhead'],
            transformer_num_layers=config['model']['transformer']['num_layers'],
            transformer_dim_feedforward=config['model']['transformer']['dim_feedforward'],
            output_dim=output_dim,
            dropout=config['model']['dropout'],
            max_len=config['data']['seq_length']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model weights
    if model_path is None:
        model_path = os.path.join(config['paths']['model_dir'], 'best_model.pth')
    
    model = load_model(model, model_path)
    model = model.to(device)
    model.eval()
    
    # Make future predictions
    future_predictions = []
    current_sequence = last_sequence.clone()
    
    for _ in range(days_ahead):
        with torch.no_grad():
            # Predict the next value
            next_pred = model(current_sequence)
            future_predictions.append(next_pred.cpu().numpy()[0])
            
            # Update the sequence for the next prediction
            # Remove the first time step and add the prediction as the last time step
            new_seq = torch.cat([
                current_sequence[:, 1:, :],
                torch.cat([current_sequence[:, -1:, 1:], next_pred.unsqueeze(1)], dim=2)
            ], dim=1)
            current_sequence = new_seq
    
    future_predictions = np.array(future_predictions)
    
    # Inverse transform predictions if needed
    if config['evaluation']['inverse_transform']:
        # Create dummy arrays with the same shape as the original data
        future_dummy = np.zeros((len(future_predictions), scaler.n_features_in_))
        
        # Put the predicted values in the first column (assuming Close price is the target)
        future_dummy[:, 0] = future_predictions.flatten()
        
        # Inverse transform
        future_inv = scaler.inverse_transform(future_dummy)[:, 0]
    else:
        future_inv = future_predictions.flatten()
    
    # Plot future predictions
    plt.figure(figsize=(12, 6))
    plt.plot(future_inv)
    plt.title(f'Future Stock Price Prediction ({days_ahead} days ahead)')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.grid(True)
    
    # Save the plot
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/future_prediction_{days_ahead}_days.png')
    plt.close()
    
    # Save predictions to CSV
    pd.DataFrame({
        'Day': range(1, days_ahead + 1),
        'Predicted_Price': future_inv
    }).to_csv(f'results/future_prediction_{days_ahead}_days.csv', index=False)
    
    return future_inv

if __name__ == "__main__":
    evaluate_model()
    predict_future(days_ahead=30)