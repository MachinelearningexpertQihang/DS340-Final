import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        """
        GRU model for time series prediction
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Number of hidden units
            num_layers (int): Number of GRU layers
            output_dim (int): Number of output features
            dropout (float): Dropout rate
        """
        super(GRUModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # GRU forward pass
        out, _ = self.gru(x, h0)
        
        # Get the output from the last time step
        out = self.fc(out[:, -1, :])
        
        return out
    
    def get_features(self, x):
        """
        Extract features from the GRU model
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Feature tensor of shape (batch_size, hidden_dim)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        return out[:, -1, :]