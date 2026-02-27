import torch
import math
from src.utils import trunc_normal_init_

class PuzzleEmbedding(torch.nn.Module):
    def __init__(self, vocab_size: int, n_puzzle_embedding_tokens: int, d_hidden: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_puzzle_embedding_tokens = n_puzzle_embedding_tokens if n_puzzle_embedding_tokens > 0 else 1
        self.d_hidden = d_hidden
        self.puzzle_embedding = torch.nn.Embedding(vocab_size, d_hidden)
        if n_puzzle_embedding_tokens > 0:
            self.sparse_latent_puzzle_embedding = SparseEmbedding(vocab_size, d_hidden, init_std=0.0)
        self._init_weights()

    def forward(self, input_ids: torch.Tensor, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        embeddings = self.puzzle_embedding(input_ids)
        
        latent_puzzle_embedding = self.sparse_latent_puzzle_embedding(puzzle_identifiers)
        pad_count = self.n_puzzle_embedding_tokens * self.d_hidden - latent_puzzle_embedding.shape[-1]
        if pad_count > 0:
            latent_puzzle_embedding = torch.nn.functional.pad(latent_puzzle_embedding, (0, pad_count))
        embeddings = torch.cat((latent_puzzle_embedding.view(-1, self.n_puzzle_embedding_tokens, self.d_hidden), embeddings), dim=-2)
        
        scale = 1 / math.sqrt(self.d_hidden)
        embeddings = embeddings * scale

        return embeddings

    def _init_weights(self):
        scale = 1 / math.sqrt(self.d_hidden)
        self.puzzle_embedding.weight = trunc_normal_init_(self.puzzle_embedding.weight, std=scale)

class SparseEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int = 768, init_std: float = 0.0):
        super().__init__()
        self.weights = torch.nn.Buffer(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std), persistent=True
        )
        self.local_weights = torch.nn.Buffer(torch.zeros(batch_size, embedding_dim, requires_grad=True), persistent=False)
        self.local_ids = torch.nn.Buffer(torch.zeros(batch_size, dtype=torch.int32), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return self.weights[inputs]
            
        with torch.no_grad():
            self.local_weights.copy_(self.weights[inputs])
            self.local_ids.copy_(inputs)

        return self.local_weights