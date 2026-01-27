import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) as described in:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    def __init__(self, dim, max_position_embeddings=512, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Build cache for positional embeddings
        self._build_cache(max_position_embeddings)
    
    def _build_cache(self, max_seq_len):
        """Build cached cos and sin embeddings."""
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)
    
    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, x):
        """
        Apply rotary positional embeddings to input embeddings.
        
        Args:
            x: Input embeddings [batch, seq_len, d_model]
        
        Returns:
            x_rot: Rotated embeddings [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        if seq_len > self.max_position_embeddings:
            # Rebuild cache if needed
            self._build_cache(seq_len)
        
        cos = self.cos_cached[:, :, :seq_len, :].to(x.device)  # [1, 1, seq_len, dim]
        sin = self.sin_cached[:, :, :seq_len, :].to(x.device)  # [1, 1, seq_len, dim]
        
        # Remove batch and head dimensions: [seq_len, dim]
        cos = cos.squeeze(0).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(0).squeeze(0)  # [seq_len, dim]
        
        # Expand to match batch: [batch, seq_len, dim]
        cos = cos.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, seq_len, dim]
        sin = sin.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, seq_len, dim]
        
        x_embed = (x * cos) + (self._rotate_half(x) * sin)
        
        return x_embed