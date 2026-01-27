import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = int(expansion * hidden_size)
        self.fc1 = nn.Linear(hidden_size, inter)
        self.fc2 = nn.Linear(inter, hidden_size)

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = self.fc2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, rotary_embedding=None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads 
        self.d_k = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        q = self.q_proj(x)  
        k = self.k_proj(x)  
        v = self.v_proj(x)  
        
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)
        output = self.out_proj(attn_output)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, use_attention=True, rotary_embedding=None):
        super().__init__()
        self.use_attention = use_attention
        
        if use_attention:
            self.attention = MultiHeadAttention(d_model, n_heads, dropout)
            self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = MLP(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        if self.use_attention:
            residual = x
            x = self.attention(x, mask)
            x = self.norm1(residual + self.dropout(x))
        
        residual = x
        x = self.ffn(x)
        x = self.norm2(residual + self.dropout(x))
        
        return x