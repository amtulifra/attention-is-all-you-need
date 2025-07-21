import torch
import torch.nn as nn
import math
from .encoder import Encoder
from .decoder import Decoder
from .positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    """
    Transformer model for sequence-to-sequence tasks.
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 max_seq_length=5000, dropout=0.1):
        """
        Initialize the Transformer model.
        
        Args:
            src_vocab_size: Size of the source vocabulary
            tgt_vocab_size: Size of the target vocabulary
            d_model: Dimension of the model
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            d_ff: Dimension of the feed-forward network
            max_seq_length: Maximum sequence length for positional encoding
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        
        # Embedding layers
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Encoder and decoder
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)
        
        # Output layer
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for embeddings
        self.scale = math.sqrt(d_model)
        
        # Initialize parameters
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src, src_mask=None, store_attention=False):
        """
        Encode the source sequence.
        
        Args:
            src: Source sequence tensor of shape [batch_size, src_len]
            src_mask: Optional source mask of shape [batch_size, 1, 1, src_len]
            store_attention: If True, returns attention weights from all encoder layers
            
        Returns:
            Encoded source sequence of shape [batch_size, src_len, d_model]
            Attention weights if store_attention=True
        """
        # Embed and add positional encoding
        x = self.encoder_embedding(src) * self.scale
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through encoder
        return self.encoder(x, src_mask, store_attention)
    
    def decode(self, tgt, memory, src_mask=None, tgt_mask=None, store_attention=False):
        """
        Decode the target sequence.
        
        Args:
            tgt: Target sequence tensor of shape [batch_size, tgt_len]
            memory: Encoder output of shape [batch_size, src_len, d_model]
            src_mask: Optional source mask of shape [batch_size, 1, 1, src_len]
            tgt_mask: Optional target mask of shape [batch_size, 1, tgt_len, tgt_len]
            store_attention: If True, returns attention weights from all decoder layers
            
        Returns:
            Decoded sequence of shape [batch_size, tgt_len, d_model]
            Tuple of (self_attention_weights, enc_dec_attention_weights) if store_attention=True
        """
        # Embed and add positional encoding
        x = self.decoder_embedding(tgt) * self.scale
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through decoder
        return self.decoder(x, memory, src_mask, tgt_mask, store_attention)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, return_attn_weights=False):
        """
        Forward pass of the Transformer model.
        
        Args:
            src: Source sequence tensor of shape [batch_size, src_len]
            tgt: Target sequence tensor of shape [batch_size, tgt_len]
            src_mask: Optional source mask of shape [batch_size, 1, 1, src_len] or [batch_size, 1, src_len]
            tgt_mask: Optional target mask of shape [batch_size, 1, tgt_len, tgt_len] or [batch_size, tgt_len, tgt_len]
            return_attn_weights: If True, returns attention weights for visualization
            
        Returns:
            Output tensor of shape [batch_size, tgt_len, tgt_vocab_size]
            If return_attn_weights is True, also returns attention weights
        """
        # Encode source sequence
        enc_output, enc_attn_weights = self.encode(src, src_mask, return_attn_weights)
        
        # Decode target sequence
        dec_output, dec_attn_weights = self.decode(tgt, enc_output, src_mask, tgt_mask, return_attn_weights)
        
        # Project to vocabulary size
        output = self.fc_out(dec_output)
        
        if return_attn_weights:
            return output, {
                'encoder_self_attention': enc_attn_weights,
                'decoder_self_attention': dec_attn_weights[0] if dec_attn_weights else None,
                'decoder_encoder_attention': dec_attn_weights[1] if dec_attn_weights else None
            }
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """
        Generate a square mask for the sequence to prevent attending to future positions.
        
        Args:
            sz: Size of the mask (sequence length)
            
        Returns:
            Mask tensor of shape [sz, sz] where positions (i, j) are True if j > i
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def count_parameters(self):
        """Return the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
