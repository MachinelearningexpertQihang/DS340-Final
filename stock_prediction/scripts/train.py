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
    train_data = torch.load(train_data_path, weights_only=False)  # 显式设置 weights_only=False
    X_train, y_train = train_data['X'], train_data['y']
    
    # Load validation data
    val_data_path = os.path.join(config['paths']['processed_data_dir'], 'val_data.pt')
    if os.path.exists(val_data_path):
        val_data = torch.load(val_data_path, weights_only=False)
        X_val, y_val = val_data['X'], val_data['y']
        # Move validation data to device
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        use_validation = True
        print(f"Validation data loaded, shape: {X_val.shape}")
    else:
        # If no validation data found, split training data
        val_size = int(len(X_train) * 0.2)  # 使用20%的训练数据作为验证集
        indices = torch.randperm(len(X_train))
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        X_val, y_val = X_train[val_indices], y_train[val_indices]
        X_train, y_train = X_train[train_indices], y_train[train_indices]
        
        # Move validation data to device
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        use_validation = True
        print(f"Validation data created from training set, shape: {X_val.shape}")
    
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
    val_losses = []
    
    # Create directory for model checkpoints
    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    
    best_val_loss = float('inf')
    
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
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
        
        # Calculate average training loss for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        if use_validation:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_losses.append(val_loss.item())
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss.item():.6f}")
            
            # Update learning rate scheduler based on validation loss
            if config['training']['use_lr_scheduler']:
                scheduler.step(val_loss.item())
            
            # Save best model based on validation loss
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                save_model(model, os.path.join(config['paths']['model_dir'], 'best_model.pth'))
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}")
            
            # Update learning rate scheduler based on training loss
            if config['training']['use_lr_scheduler']:
                scheduler.step(avg_train_loss)
            
            # Save best model based on training loss
            if avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                save_model(model, os.path.join(config['paths']['model_dir'], 'best_model.pth'))
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            save_model(model, os.path.join(config['paths']['model_dir'], f'model_epoch_{epoch+1}.pth'))
    
    # Save final model
    save_model(model, os.path.join(config['paths']['model_dir'], 'final_model.pth'))
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    if use_validation:
        plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_validation_loss.png')
    plt.close()
    
    return model

if __name__ == "__main__":
    train_model()