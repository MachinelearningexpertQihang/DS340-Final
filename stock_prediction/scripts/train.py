import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from models.gru import GRUModel
from models.transformer import TransformerModel
from models.gru_transformer import GRUTransformerModel
from scripts.utils import load_config, save_model

def train_model(config_path='config.yaml'):
    """
    Train the model
    
    Args:
        config_path (str): Path to config file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config['training']['use_gpu'] else 'cpu')
    print(f"Using device: {device}")
    
    # Load processed data
    train_data_path = os.path.join(config['paths']['processed_data_dir'], 'train_data.pt')
    train_data = torch.load(train_data_path)
    X_train, y_train = train_data['X'], train_data['y']
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True
    )
    
    # Get input dimensions
    input_dim = X_train.shape[2]  # Number of features
    output_dim = y_train.shape[1]  # Number of output dimensions
    
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
    
    # Move model to device
    model = model.to(device)
    
    # Initialize loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Learning rate scheduler
    if config['training']['use_lr_scheduler']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    train_losses = []
    
    # Create directory for model checkpoints
    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    
    best_loss = float('inf')
    
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimization
            loss.backward()
            
            # Gradient clipping
            if config['training']['clip_grad_norm']:
                nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                
            optimizer.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': epoch_loss / (batch_idx + 1)})
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Update learning rate scheduler
        if config['training']['use_lr_scheduler']:
            scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, os.path.join(config['paths']['model_dir'], 'best_model.pth'))
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            save_model(model, os.path.join(config['paths']['model_dir'], f'model_epoch_{epoch+1}.pth'))
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    # Save final model
    save_model(model, os.path.join(config['paths']['model_dir'], 'final_model.pth'))
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_loss.png')
    plt.close()
    
    return model

if __name__ == "__main__":
    train_model()