import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        """
        Positional encoding for transformer model
        
        Args:
            d_model (int): Embedding dimension
            max_len (int): Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but should be saved and moved with the model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input tensor
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, dropout=0.1, max_len=1000):
        """
        Transformer model for time series prediction
        
        Args:
            input_dim (int): Number of input features
            d_model (int): Embedding dimension
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dim_feedforward (int): Dimension of feedforward network
            output_dim (int): Number of output features
            dropout (float): Dropout rate
            max_len (int): Maximum sequence length
        """
        super(TransformerModel, self).__init__()
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layer
        self.fc = nn.Linear(d_model, output_dim)
        
        self.d_model = d_model
        
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            mask (torch.Tensor, optional): Mask tensor
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Input embedding
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Get the output from the last time step
        x = x[:, -1, :]
        
        # Output layer
        x = self.fc(x)
        
        return x
    
    def get_features(self, x, mask=None):
        """
        Extract features from the transformer model
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            mask (torch.Tensor, optional): Mask tensor
            
        Returns:
            torch.Tensor: Feature tensor of shape (batch_size, d_model)
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        return x[:, -1, :]