import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        """
        Args:
            d_model: Dimension of the model
            d_ff: Dimension of the feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
