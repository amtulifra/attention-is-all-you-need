import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    """
    A single layer of the decoder.
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
        # Self attention layer
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Encoder-decoder attention layer
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None, store_attention=False):
        """
        Args:
            x: Input tensor of shape [batch_size, tgt_seq_len, d_model]
            enc_output: Encoder output of shape [batch_size, src_seq_len, d_model]
            src_mask: Optional mask for encoder output of shape [batch_size, 1, 1, src_seq_len]
            tgt_mask: Optional mask for target sequence of shape [batch_size, 1, tgt_seq_len, tgt_seq_len]
            store_attention: If True, returns attention weights
            
        Returns:
            Output tensor of shape [batch_size, tgt_seq_len, d_model]
            Tuple of (self_attention_weights, enc_dec_attention_weights) if store_attention=True
        """
        # Self attention with residual connection and layer normalization
        self_attn_output, self_attn_weights = self.self_attn(x, x, x, tgt_mask, store_attention)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Encoder-decoder attention with residual connection and layer normalization
        enc_dec_attn_output, enc_dec_attn_weights = self.enc_dec_attn(
            x, enc_output, enc_output, src_mask, store_attention
        )
        x = self.norm2(x + self.dropout(enc_dec_attn_output))
        
        # Feed-forward network with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        if store_attention:
            return x, (self_attn_weights, enc_dec_attn_weights)
        return x, None


class Decoder(nn.Module):
    """
    Transformer Decoder consisting of multiple decoder layers.
    """
    def __init__(self, num_layers=6, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        """
        Args:
            num_layers: Number of decoder layers
            d_model: Dimension of the model
            num_heads: Number of attention heads
            d_ff: Dimension of the feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None, store_attention=False):
        """
        Args:
            x: Input tensor of shape [batch_size, tgt_seq_len, d_model]
            enc_output: Encoder output of shape [batch_size, src_seq_len, d_model]
            src_mask: Optional mask for encoder output of shape [batch_size, 1, 1, src_seq_len]
            tgt_mask: Optional mask for target sequence of shape [batch_size, 1, tgt_seq_len, tgt_seq_len]
            store_attention: If True, returns attention weights from all layers
            
        Returns:
            Output tensor of shape [batch_size, tgt_seq_len, d_model]
            Tuple of (all_self_attn_weights, all_enc_dec_attn_weights) if store_attention=True
        """
        all_self_attn_weights = []
        all_enc_dec_attn_weights = []
        
        for layer in self.layers:
            x, attn_weights = layer(x, enc_output, src_mask, tgt_mask, store_attention)
            if store_attention:
                self_attn, enc_dec_attn = attn_weights
                all_self_attn_weights.append(self_attn)
                all_enc_dec_attn_weights.append(enc_dec_attn)
        
        if store_attention:
            return x, (all_self_attn_weights, all_enc_dec_attn_weights)
        return x, None
