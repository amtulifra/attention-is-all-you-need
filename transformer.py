import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"Model dimension ({d_model}) must be divisible by number of heads ({num_heads})")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        self.attention_weights = None
    
    def forward(self, q, k, v, mask=None, store_attention=False):
        batch_size = q.size(0)
        
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        if store_attention:
            self.attention_weights = attn.detach()
        
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, store_attention=False):
        attn_output = self.self_attn(x, x, x, mask, store_attention)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None, store_attention=False):
        self_attn_output = self.self_attn(x, x, x, tgt_mask, store_attention)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        enc_dec_attn_output = self.enc_dec_attn(x, enc_output, enc_output, src_mask, store_attention)
        x = self.norm2(x + self.dropout(enc_dec_attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 max_seq_length=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src, src_mask=None, store_attention=False):
        x = self.encoder_embedding(src)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask, store_attention)
            
        return x
    
    def decode(self, tgt, memory, src_mask=None, tgt_mask=None, store_attention=False):
        x = self.decoder_embedding(tgt)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask, store_attention)
            
        return self.fc_out(x)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, return_attn_weights=False):
        """Forward pass of the Transformer model.
        
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
        # Ensure masks have the right shape
        if src_mask is not None and len(src_mask.shape) == 3:
            src_mask = src_mask.unsqueeze(1)  # [batch_size, 1, 1, src_len]
            
        if tgt_mask is not None and len(tgt_mask.shape) == 3:
            tgt_mask = tgt_mask.unsqueeze(1)  # [batch_size, 1, tgt_len, tgt_len]
        
        # Encode source
        memory = self.encode(src, src_mask, store_attention=return_attn_weights)
        
        # Decode to target
        output = self.decode(tgt, memory, src_mask, tgt_mask, store_attention=return_attn_weights)
        
        if return_attn_weights:
            # Get all attention weights
            encoder_attn = getattr(self, 'encoder_attention_weights', None)
            decoder_self_attn = getattr(self, 'decoder_self_attention_weights', None)
            decoder_cross_attn = getattr(self, 'decoder_cross_attention_weights', None)
            
            # Clean up attributes
            if hasattr(self, 'encoder_attention_weights'):
                del self.encoder_attention_weights
            if hasattr(self, 'decoder_self_attention_weights'):
                del self.decoder_self_attention_weights
            if hasattr(self, 'decoder_cross_attention_weights'):
                del self.decoder_cross_attention_weights
            
            return output, (encoder_attn, decoder_self_attn, decoder_cross_attn)
            
        return output
