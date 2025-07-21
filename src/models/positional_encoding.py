import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Positional Encoding module that adds positional information to input embeddings.
    """
    def __init__(self, d_model=512, max_len=5000):
        """
        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
        """
        super().__init__()
        self.d_model = d_model
        
        # Create positional encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor with positional encodings added
        """
        # Add positional encodings to input embeddings
        x = x * math.sqrt(self.d_model)  # Scale embeddings
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x
