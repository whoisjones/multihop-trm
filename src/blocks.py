import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-5):
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
        intermediate_size = round(up_proj_factor * hidden_size * 2 / 3)
        alignment = 256
        intermediate_size = (-(intermediate_size // -alignment)) * alignment
        self.up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class MLPMixer(nn.Module):
    def __init__(
        self, 
        puzzle_len: int, 
        n_puzzle_embedding_tokens: int, 
        hidden_size: int, 
        dropout: float = 0.0, 
        **kwargs
    ):
        super().__init__()
        self.token_mixing = SwiGLU(puzzle_len + n_puzzle_embedding_tokens)
        self.rms_norm = RMSNorm()
        self.channel_mixing = SwiGLU(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        hidden = inputs.transpose(1, 2)
        hidden = hidden + self.token_mixing(self.rms_norm(hidden))
        hidden = hidden.transpose(1, 2)
        hidden = self.dropout(hidden)
        hidden = hidden + self.channel_mixing(self.rms_norm(hidden))
        hidden = self.dropout(hidden)
        return hidden