import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    """
    A single layer of the encoder.
    """
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        """
        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            d_ff: Dimension of the feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, store_attention=False):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional mask tensor of shape [batch_size, 1, 1, seq_len]
            store_attention: If True, returns attention weights
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
            Attention weights if store_attention=True
        """
        # Self attention with residual connection and layer normalization
        attn_output, attn_weights = self.self_attn(x, x, x, mask, store_attention)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward network with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        if store_attention:
            return x, attn_weights
        return x, None


class Encoder(nn.Module):
    """
    Transformer Encoder consisting of multiple encoder layers.
    """
    def __init__(self, num_layers=6, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        """
        Args:
            num_layers: Number of encoder layers
            d_model: Dimension of the model
            num_heads: Number of attention heads
            d_ff: Dimension of the feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None, store_attention=False):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional mask tensor of shape [batch_size, 1, 1, seq_len]
            store_attention: If True, returns attention weights from all layers
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
            List of attention weights if store_attention=True
        """
        attention_weights = []
        
        for layer in self.layers:
            x, attn = layer(x, mask, store_attention)
            if store_attention:
                attention_weights.append(attn)
        
        if store_attention:
            return x, attention_weights
        return x, None
