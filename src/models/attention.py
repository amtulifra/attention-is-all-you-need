import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module
    """
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"Model dimension ({d_model}) must be divisible by number of heads ({num_heads})")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, q, k, v, mask=None, store_attention=False):
        """
        Args:
            q: Query tensor of shape [batch_size, seq_len_q, d_model]
            k: Key tensor of shape [batch_size, seq_len_kv, d_model]
            v: Value tensor of shape [batch_size, seq_len_kv, d_model]
            mask: Optional mask tensor of shape [batch_size, 1, 1, seq_len_kv] or [batch_size, seq_len_q, seq_len_kv]
            store_attention: If True, returns attention weights
            
        Returns:
            output: Output tensor of shape [batch_size, seq_len_q, d_model]
            attention_weights: Optional attention weights if store_attention=True
        """
        # Ensure inputs are tensors, not tuples
        if isinstance(q, tuple):
            q = q[0]
        if isinstance(k, tuple):
            k = k[0]
        if isinstance(v, tuple):
            v = v[0]
            
        batch_size = q.size(0)
        
        # Linear projections and split into heads
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len_q, d_k]
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len_kv, d_k]
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len_kv, d_k]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch_size, num_heads, seq_len_q, seq_len_kv]
        
        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, v)  # [batch_size, num_heads, seq_len_q, d_k]
        
        # Concatenate heads and apply final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # [batch_size, seq_len_q, d_model]
        output = self.W_o(output)
        
        if store_attention:
            return output, attention_weights
        return output, None
