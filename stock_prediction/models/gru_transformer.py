import torch
import torch.nn as nn
from models.gru import GRUModel
from models.transformer import TransformerModel

class GRUTransformerModel(nn.Module):
    def __init__(self, input_dim, gru_hidden_dim, gru_num_layers, 
                 transformer_d_model, transformer_nhead, transformer_num_layers, 
                 transformer_dim_feedforward, output_dim, dropout=0.1, max_len=1000):
        """
        Combined GRU and Transformer model for stock prediction
        
        Args:
            input_dim (int): Number of input features
            gru_hidden_dim (int): Number of hidden units in GRU
            gru_num_layers (int): Number of GRU layers
            transformer_d_model (int): Embedding dimension for transformer
            transformer_nhead (int): Number of attention heads
            transformer_num_layers (int): Number of transformer layers
            transformer_dim_feedforward (int): Dimension of feedforward network
            output_dim (int): Number of output features
            dropout (float): Dropout rate
            max_len (int): Maximum sequence length
        """
        super(GRUTransformerModel, self).__init__()
        
        # GRU model
        self.gru_model = GRUModel(
            input_dim=input_dim,
            hidden_dim=gru_hidden_dim,
            num_layers=gru_num_layers,
            output_dim=output_dim,
            dropout=dropout
        )
        
        # Transformer model
        self.transformer_model = TransformerModel(
            input_dim=input_dim,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
            dim_feedforward=transformer_dim_feedforward,
            output_dim=output_dim,
            dropout=dropout,
            max_len=max_len
        )
        
        # Fusion layer
        self.fusion = nn.Linear(gru_hidden_dim + transformer_d_model, output_dim)
        
        # Loss function for validation
        self.criterion = nn.MSELoss()
        
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            mask (torch.Tensor, optional): Mask tensor for transformer
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Get features from GRU
        gru_features = self.gru_model.get_features(x)
        
        # Get features from Transformer
        transformer_features = self.transformer_model.get_features(x, mask)
        
        # Concatenate features
        combined_features = torch.cat((gru_features, transformer_features), dim=1)
        
        # Fusion layer
        output = self.fusion(combined_features)
        
        return output
    
    def calculate_loss(self, predictions, targets):
        """
        Calculate loss for training or validation
        
        Args:
            predictions (torch.Tensor): Predicted values from the model
            targets (torch.Tensor): Target values
            
        Returns:
            torch.Tensor: Calculated loss
        """
        return self.criterion(predictions, targets)
    
    def validate(self, x, targets, mask=None):
        """
        Run validation and return loss
        
        Args:
            x (torch.Tensor): Input tensor
            targets (torch.Tensor): Target values
            mask (torch.Tensor, optional): Mask for transformer
            
        Returns:
            torch.Tensor: Validation loss
        """
        # Set model to evaluation mode
        self.eval()
        
        with torch.no_grad():
            # Get predictions
            predictions = self(x, mask)
            
            # Calculate validation loss
            val_loss = self.calculate_loss(predictions, targets)
            
        # Set model back to training mode
        self.train()
        
        return val_loss