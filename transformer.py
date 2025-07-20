import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-head attention that enables the model to focus on different parts of the input sequence."""
    
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"Model dimension ({d_model}) must be divisible by number of heads ({num_heads})")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Split into multiple heads
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention and combine heads
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(x)

class PositionWiseFeedForward(nn.Module):
    """Two-layer feed-forward network with ReLU activation and dropout."""
    
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class PositionalEncoding(nn.Module):
    """Adds position information to input embeddings using sine and cosine functions."""
    
    def __init__(self, d_model=512, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encodings
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))  # [d_model/2]
        
        # Initialize positional encoding matrix
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        
        # Fill in the positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        
        # Register as buffer (no batch dimension)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        # Get the sequence length from the input
        seq_len = x.size(1)
        
        # Add positional encoding to input
        # pe is [max_len, d_model], we take first 'seq_len' elements
        # and unsqueeze to [1, seq_len, d_model] for broadcasting
        x = x + self.pe[:seq_len].unsqueeze(0).to(x.device)
        return x

class EncoderLayer(nn.Module):
    """A single encoder layer with self-attention and feed-forward network."""
    
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn))
        
        # Feed forward with residual connection and layer norm
        ff = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff))
        
        return x

class DecoderLayer(nn.Module):
    """A single decoder layer with self-attention, encoder-decoder attention, and feed-forward network."""
    
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn))
        
        # Encoder-decoder attention
        attn = self.enc_dec_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn))
        
        # Feed forward
        ff = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff))
        
        return x

class Transformer(nn.Module):
    """The Transformer model from 'Attention Is All You Need' paper."""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 max_seq_length=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Embedding layers
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Encoder and decoder stacks
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Final output layer
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for embeddings
        self.scale = math.sqrt(d_model)
        
        # Initialize parameters
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize weights with xavier_uniform and set biases to zero"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                nn.init.constant_(p, 0.0)
    
    def encode(self, src, src_mask=None):
        """
        Encodes the source sequence.
        
        Args:
            src: Input tensor of shape [batch_size, src_len]
            src_mask: Optional tensor of shape [batch_size, 1, 1, src_len]
            
        Returns:
            Tensor of shape [batch_size, src_len, d_model]
        """
        # Input embeddings and scaling
        x = self.encoder_embedding(src)  # [batch_size, src_len] -> [batch_size, src_len, d_model]
        x = x * self.scale
        
        # Add positional encoding
        x = self.pos_encoding(x)  # [batch_size, src_len, d_model]
        x = self.dropout(x)
        
        # Create source mask if not provided
        if src_mask is None:
            src_mask = (src != 2).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
            
        return x
        
    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        """
        Decodes the target sequence using the encoder output.
        
        Args:
            tgt: Target tensor of shape [batch_size, tgt_len]
            memory: Encoder output of shape [batch_size, src_len, d_model]
            src_mask: Optional source mask of shape [batch_size, 1, 1, src_len]
            tgt_mask: Optional target mask of shape [batch_size, 1, tgt_len, tgt_len]
            
        Returns:
            Tensor of shape [batch_size, tgt_len, tgt_vocab_size]
        """
        # Input embeddings and scaling
        x = self.decoder_embedding(tgt)  # [batch_size, tgt_len] -> [batch_size, tgt_len, d_model]
        x = x * self.scale
        
        # Add positional encoding
        x = self.pos_encoding(x)  # [batch_size, tgt_len, d_model]
        x = self.dropout(x)
        
        # Create target mask if not provided
        if tgt_mask is None:
            # Create padding mask
            tgt_pad_mask = (tgt != 2).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, tgt_len]
            
            # Create look-ahead mask
            seq_len = tgt.size(1)
            nopeak_mask = (1 - torch.triu(
                torch.ones((1, 1, seq_len, seq_len), device=tgt.device), 
                diagonal=1
            )).bool()
            
            # Combine masks [batch_size, 1, tgt_len, tgt_len]
            tgt_mask = tgt_pad_mask & nopeak_mask
        
        # Ensure src_mask is 4D [batch_size, 1, 1, src_len]
        if src_mask is not None and src_mask.dim() == 3:
            src_mask = src_mask.unsqueeze(1)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
            
        # Final linear projection
        return self.fc_out(x)  # [batch_size, tgt_len, tgt_vocab_size]
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Run the full encoder-decoder pipeline.
        
        Args:
            src: Source sequence tensor of shape [batch_size, src_len]
            tgt: Target sequence tensor of shape [batch_size, tgt_len]
            src_mask: Optional source mask of shape [batch_size, 1, 1, src_len] or [batch_size, 1, src_len]
            tgt_mask: Optional target mask of shape [batch_size, 1, tgt_len, tgt_len] or [batch_size, tgt_len, tgt_len]
            
        Returns:
            Output tensor of shape [batch_size, tgt_len, tgt_vocab_size]
        """
        # Ensure inputs are on the correct device
        device = next(self.parameters()).device
        src = src.to(device)
        tgt = tgt.to(device)
        
        # Handle source mask
        if src_mask is not None:
            src_mask = src_mask.to(device)
            # Ensure src_mask is 4D [batch_size, 1, 1, src_len]
            if src_mask.dim() == 3:
                src_mask = src_mask.unsqueeze(1)
        
        # Handle target mask
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(device)
            # Ensure tgt_mask is 4D [batch_size, 1, tgt_len, tgt_len]
            if tgt_mask.dim() == 3:
                tgt_mask = tgt_mask.unsqueeze(1)
        
        # Encode the source sequence
        memory = self.encode(src, src_mask)
        
        # Decode the target sequence
        output = self.decode(tgt, memory, src_mask, tgt_mask)
        
        return output
