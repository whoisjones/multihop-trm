import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, up_proj_factor: int = 4):
        super().__init__()
        up_proj_dim = round(up_proj_factor * hidden_size * 2 / 3)
        alignment = 256
        up_proj_dim = (-(up_proj_dim // -alignment)) * alignment
        self.up_proj = nn.Linear(hidden_size, up_proj_dim * 2, bias=False)
        self.down_proj = nn.Linear(up_proj_dim, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class MLPBlock(nn.Module):
    def __init__(self, puzzle_len: int, hidden_size: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.proj1 = SwiGLU(puzzle_len)
        self.rms_norm = RMSNorm()
        self.proj2 = SwiGLU(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        hidden = inputs.transpose(1, 2)
        out = self.proj1(hidden)
        hidden = self.rms_norm(hidden + self.dropout(out))
        hidden = hidden.transpose(1, 2)

        out = self.proj2(hidden)
        hidden = self.rms_norm(hidden + self.dropout(out))
        return hidden


class TransformerBlock(nn.Module):
    def __init__(self, d_hidden, n_heads, dropout=0.1, **kwargs):
        super().__init__()
        self.attention = MultiHeadAttention(d_hidden, n_heads, dropout)
        self.norm = RMSNorm(d_hidden)
        self.proj = SwiGLU(d_hidden)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs):
        hidden = self.norm(inputs + self.dropout(self.attention(inputs)))
        hidden = self.norm(hidden + self.dropout(self.proj(hidden)))
        return hidden


class MultiHeadAttention(nn.Module):
    def __init__(self, d_hidden, n_heads, dropout=0.1, rotary_embedding=None):
        super().__init__()
        assert d_hidden % n_heads == 0, "d_hidden must be divisible by n_heads"
        
        self.d_hidden = d_hidden
        self.n_heads = n_heads 
        self.d_k = d_hidden // n_heads
        
        self.q_proj = nn.Linear(d_hidden, d_hidden)
        self.k_proj = nn.Linear(d_hidden, d_hidden)
        self.v_proj = nn.Linear(d_hidden, d_hidden)
        
        self.out_proj = nn.Linear(d_hidden, d_hidden)
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
