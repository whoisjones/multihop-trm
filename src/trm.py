import math
from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn as nn

from src.loss import StablemaxCrossEntropyLoss

from src.blocks import MLPMixer
from src.puzzle_embedding import PuzzleEmbedding
from src.utils import trunc_normal_init_

@dataclass
class Metrics:
    mask: torch.Tensor
    predictions: torch.Tensor
    token_correct: torch.Tensor
    seq_correct: torch.Tensor
    loss_divisor: torch.Tensor
    valid_metrics: torch.Tensor = None


@dataclass
class TRMOutput:
    loss: torch.Tensor
    seq_loss: torch.Tensor
    q_halt_loss: torch.Tensor
    logits: torch.Tensor
    q_halt_logits: torch.Tensor

@dataclass
class TRMCarry:
    y: torch.Tensor
    z: torch.Tensor
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: dict[str, torch.Tensor]


class TRM(nn.Module):
    def __init__(
        self,
        backbone: str,
        vocab_size: int,
        puzzle_len: int,
        d_hidden: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        n_puzzle_embedding_tokens: int = 16,
        deep_supervision_steps: int = 16,
        deep_supervision_exploration_prob: float = 0.1,
        n_reasoning_steps: int = 3,
        n_latent_steps: int = 6,
        seq_loss_weight: float = 1.0,
        q_halt_loss_weight: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.d_hidden = d_hidden
        self.n_puzzle_embedding_tokens = n_puzzle_embedding_tokens
        self.deep_supervision_steps = deep_supervision_steps
        self.deep_supervision_exploration_prob = deep_supervision_exploration_prob
        self.n_reasoning_steps = n_reasoning_steps
        self.n_latent_steps = n_latent_steps
        self.seq_loss_weight = seq_loss_weight
        self.q_halt_loss_weight = q_halt_loss_weight

        self.embedding_layer = PuzzleEmbedding(vocab_size, n_puzzle_embedding_tokens, d_hidden)
        self.output_proj = nn.Linear(d_hidden, vocab_size, bias=False)
        self.q_proj = nn.Linear(d_hidden, 1, bias=True)
        
        self.reasoning_blocks = nn.Sequential(
            *[MLPMixer(puzzle_len, n_puzzle_embedding_tokens, d_hidden) for _ in range(n_layers)],
            nn.LayerNorm(d_hidden, bias=False)
        )

        self.answer_init = nn.Buffer(trunc_normal_init_(torch.empty(self.d_hidden), std=1), persistent=True)
        self.latent_init = nn.Buffer(trunc_normal_init_(torch.empty(self.d_hidden), std=1), persistent=True)

        self.seq_loss_fn = StablemaxCrossEntropyLoss()
        self.q_halt_loss_fn = nn.BCEWithLogitsLoss()

        self._init_weights()
        
    def _init_weights(self):
        scale = 1 / math.sqrt(self.d_hidden)
        self.embedding_layer.puzzle_embedding.weight = trunc_normal_init_(self.embedding_layer.puzzle_embedding.weight, std=scale)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        with torch.no_grad():
            self.q_proj.weight.zero_()
            self.q_proj.bias.fill_(-5)

    def init_carry(self, batch):
        batch_size, puzzle_len = batch['inputs'].shape
        device = batch['inputs'].device

        carry = TRMCarry(
            y=torch.empty(batch_size, puzzle_len + self.n_puzzle_embedding_tokens, self.d_hidden, device=device),
            z=torch.empty(batch_size, puzzle_len + self.n_puzzle_embedding_tokens, self.d_hidden, device=device),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v, device=device) for k, v in batch.items()}
        )
        return carry

    def update_carry(self, old_carry, batch):
        new_carry = TRMCarry(
            y=torch.where(old_carry.halted.view(-1, 1, 1), self.answer_init, old_carry.y),
            z=torch.where(old_carry.halted.view(-1, 1, 1), self.latent_init, old_carry.z),
            steps=torch.where(old_carry.halted, 0, old_carry.steps),
            halted=old_carry.halted,
            current_data={k: torch.where(old_carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v) for k, v in old_carry.current_data.items()}
        )
        return new_carry

    @torch.no_grad()
    def compute_metrics(self, carry: TRMCarry, logits: torch.Tensor) -> Metrics:
        mask = (carry.current_data['labels'] != -100)
        predictions = logits.argmax(dim=-1)
        token_correct = predictions == carry.current_data['labels']
        return Metrics(
            mask=mask,
            loss_divisor=mask.sum(-1).clamp_min(1).unsqueeze(-1),
            predictions=predictions,
            token_correct=token_correct,
            seq_correct=((mask & (token_correct)).sum(dim=-1) == mask.sum(dim=-1)),
        )

    def refine(self, hidden, injection):
        hidden = hidden + injection
        hidden = self.reasoning_blocks(hidden)
        return hidden

    def forward(self, batch, carry):
        batch_size = batch['inputs'].shape[0]
        carry = self.update_carry(carry, batch)

        x = self.embedding_layer(carry.current_data['inputs'], carry.current_data['puzzle_id'])
        y = carry.y
        z = carry.z

        for _reasoning_step in range(1, self.n_reasoning_steps + 1):
            is_last_step = _reasoning_step == self.n_reasoning_steps
            context = torch.no_grad if not is_last_step else nullcontext
            with context():
                for _latent_step in range(self.n_latent_steps):
                    z = self.refine(z, y + x)
                y = self.refine(y, z)

        logits = self.output_proj(y)[:, self.n_puzzle_embedding_tokens:]
        q_logits = self.q_proj(y[:, 0]).squeeze(-1)

        metrics = self.compute_metrics(carry, logits)
        seq_loss = (self.seq_loss_fn(logits, carry.current_data['labels'], metrics.mask) / metrics.loss_divisor).sum() / batch_size
        q_halt_loss = self.q_halt_loss_fn(q_logits, metrics.seq_correct.to(q_logits.dtype)).sum()

        loss = (self.seq_loss_weight * seq_loss + self.q_halt_loss_weight * q_halt_loss) 

        steps = carry.steps
        with torch.no_grad():
            steps = steps + 1
            is_last_step = steps >= self.deep_supervision_steps
            halted = is_last_step
            if self.training and (self.deep_supervision_steps > 1):
                halted = halted | (q_logits > 0)
                min_halt_steps = (torch.rand_like(q_logits) < self.deep_supervision_exploration_prob) * torch.randint_like(steps, low=2, high=self.deep_supervision_steps + 1)
                halted = halted & (steps >= min_halt_steps)

        metrics.valid_metrics = halted & (metrics.mask.sum(-1) > 0)

        output = TRMOutput(
            loss=loss,
            seq_loss=seq_loss.detach(),
            q_halt_loss=q_halt_loss.detach(),
            logits=logits,
            q_halt_logits=q_logits,
        )

        carry = TRMCarry(
            y=y.detach(),
            z=z.detach(),
            steps=steps,
            halted=halted,
            current_data=carry.current_data,
        )

        return output, carry, metrics
