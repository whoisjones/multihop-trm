import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_scheduler
from adam_atan2_pytorch import AdamAtan2
from .blocks import TransformerBlock, MLPBlock
from src.utils import trunc_normal_init_, PRECISION_MAP

class TRM(nn.Module):
    def __init__(
        self,
        backbone,
        vocab_size,
        puzzle_len,
        deep_supervision_steps,
        deep_supervision_exploration_prob,
        d_hidden,
        n_heads,
        n_layers,
        n_reasoning_steps,
        n_latent_steps,
        tie_embeddings,
        dropout,
    ):
        super().__init__()
        
        self.d_hidden = d_hidden
        self.deep_supervision_steps = deep_supervision_steps
        self.deep_supervision_exploration_prob = deep_supervision_exploration_prob
        self.n_reasoning_steps = n_reasoning_steps
        self.n_latent_steps = n_latent_steps

        self.embedding_layer = nn.Embedding(vocab_size, d_hidden)
        self.output_proj = nn.Linear(d_hidden, vocab_size, bias=False)
        self.q_proj = nn.Linear(d_hidden, 2, bias=True)
        if tie_embeddings:
            self.output_proj.weight = self.embedding_layer.weight
        
        if backbone == "transformer":
            self.reasoning_blocks = nn.Sequential(*[
                TransformerBlock(d_hidden, n_heads, dropout)
                for _ in range(n_layers)
            ])
        elif backbone == "mlp":
            self.reasoning_blocks = nn.Sequential(*[
                MLPBlock(puzzle_len, d_hidden, dropout)
                for _ in range(n_layers)
            ])
        else:
            raise ValueError(f"Invalid backbone: {backbone}")

        self.answer_init = nn.Buffer(trunc_normal_init_(torch.empty(self.d_hidden), std=1), persistent=True)
        self.latent_init = nn.Buffer(trunc_normal_init_(torch.empty(self.d_hidden), std=1), persistent=True)
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        with torch.no_grad():
            self.q_proj.weight.zero_()
            self.q_proj.bias.fill_(-5)

    def reset_answer_and_latent(self, halted_flag, answer_tensor, latent_tensor):
        reset_answer = torch.where(halted_flag.view(-1, 1, 1), self.answer_init, answer_tensor)
        reset_latent = torch.where(halted_flag.view(-1, 1, 1), self.latent_init, latent_tensor)
        return reset_answer, reset_latent

    def forward(self, batch, carry):
        y, z = self.reset_answer_and_latent(carry['halted'], carry['answer'], carry['latent'])
        steps = torch.where(carry['halted'], 0, carry['steps'])
        current_data = {
            k: torch.where(
                carry['halted'].view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v
            )
            for k, v in carry['current_data'].items()
        }

        x = self.embedding_layer(current_data['inputs'])
        with torch.no_grad():
            for _reasoning_step in range(self.n_reasoning_steps - 1):
                for _latent_step in range(self.n_latent_steps):
                    z = self.forward_pass(z, y + x)
                y = self.forward_pass(y, z)
        
        for _latent_step in range(self.n_reasoning_steps):
            z = self.forward_pass(z, x + y)
        y = self.forward_pass(y, z)

        logits = self.output_proj(y)

        q_input = y.mean(dim=1)
        q_logits = self.q_proj(q_input).to(torch.float32)
        q_halt_logits, q_continue_logits = (q_logits[..., 0], q_logits[..., 1])

        with torch.no_grad():
            steps = steps + 1
            is_last_step = steps >= self.deep_supervision_steps
            halted = is_last_step
            if self.training and (self.deep_supervision_steps > 1):
                halted = halted | (q_halt_logits > 0)
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.deep_supervision_exploration_prob) * torch.randint_like(steps, low=2, high=self.deep_supervision_steps + 1)
                halted = halted & (steps >= min_halt_steps)

        logits_output = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        carry_output = {
            "answer": y.detach(),
            "latent": z.detach(),
            "steps": steps,
            "halted": halted,
            "current_data": current_data,
        }

        return logits_output, carry_output
    
    def forward_pass(self, hidden, injection):
        hidden = hidden + injection
        output = self.reasoning_blocks(hidden)
        return output
    
    def generate(self, question_ids, max_length=50, latent_len=32, temperature=1.0):
        batch_size = question_ids.size(0)
        device = question_ids.device
        
        # Start with a beginning-of-sequence token (or zeros)
        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        for i in range(max_length):
            # Get predictions for current sequence
            logits = self.forward(question_ids, generated, latent_len)
            
            # Sample next token (with temperature for randomness)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Optional: stop if end-of-sequence token
            # if (next_token == eos_token_id).all():
            #     break
        
        return generated
    
    def configure_optimizers(
        self,
        optimizer_type,
        scheduler_type,
        learning_rate=1e-4,
        warmup_steps=1000,
        total_steps=10000,
        weight_decay=0.01
    ):
        if optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        elif optimizer_type.lower() == "adam_atan2":
            optimizer = AdamAtan2(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}. Choose from: adamw, adam, sgd")
        
        if total_steps is None:
            scheduler = None
        else:
            scheduler = get_scheduler(
                name=scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        
        return optimizer, scheduler