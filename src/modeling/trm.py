import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_scheduler
from .attention import TransformerBlock
from .embedding import RotaryEmbedding

class TRM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        n_layers=4,
        max_seq_len=512,
        dropout=0.1,
        n_reasoning_steps=8,
        n_refinement_steps=16,
        use_attention=True,
        tie_embeddings=True,
        rotary_base=10000,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_reasoning_steps = n_reasoning_steps
        self.n_refinement_steps = n_refinement_steps
        self.use_attention = use_attention
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Rotary Positional Embeddings (RoPE) - optional
        # Rotary embeddings are applied to input embeddings before transformer blocks
        if rotary_base is not None:
            self.rotary_embedding = RotaryEmbedding(
                dim=d_model,  # Full embedding dimension
                max_position_embeddings=max_seq_len,
                base=rotary_base,
            )
        else:
            self.rotary_embedding = None
        
        self.embedding_dropout = nn.Dropout(dropout)
        self.reverse_embedding = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.reverse_embedding.weight = self.token_embedding.weight
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, use_attention, None)
            for _ in range(n_layers)
        ])
        
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
    
    def embed_tokens(self, token_ids):
        batch_size, seq_len = token_ids.shape
        token_emb = self.token_embedding(token_ids)
        
        if self.rotary_embedding is not None:
            token_emb = self.rotary_embedding(token_emb)
        
        embeddings = self.embedding_dropout(token_emb)
        return embeddings
    
    def apply_transformer_blocks(self, x, mask=None):
        for block in self.transformer_blocks:
            x = block(x, mask)
        return x
    
    def forward_pass(self, x, y, z, mask=None):
        """
        Forward pass with all three streams (x, y, z).
        Used for updating z: z = net(x, y, z)
        """
        len_x = x.size(1) if x is not None else 0
        len_y = y.size(1)
        len_z = z.size(1)
        
        # Concatenate along sequence dimension
        if x is not None:
            combined = torch.cat([x, y, z], dim=1)
        else:
            combined = torch.cat([y, z], dim=1)
        
        # Pass through all transformer blocks
        combined = self.apply_transformer_blocks(combined, mask)
        
        # Split back into streams
        if x is not None:
            x_new = combined[:, :len_x, :]
            y_new = combined[:, len_x:len_x + len_y, :]
            z_new = combined[:, len_x + len_y:, :]
            return x_new, y_new, z_new
        else:
            y_new = combined[:, :len_y, :]
            z_new = combined[:, len_y:, :]
            return y_new, z_new
    
    def recursive_reasoning(self, x, y, z, mask=None, return_trajectory=False):
        """
        Deep recursion as per original implementation:
        - H_cycles-1 iterations without gradients (refinement steps)
        - 1 iteration with gradients
        
        Where:
        - H_cycles = n_refinement_steps (outer loop, refinement steps)
        - L_cycles = n_reasoning_steps (inner loop, reasoning steps)
        - z_H = y (high-level/answer stream)
        - z_L = z (low-level/reasoning stream)
        - input_embeddings = x (question stream)
        
        Structure matches original:
        for _H_step in range(H_cycles-1):  # without grad
            for _L_step in range(L_cycles):
                z_L = L_level(z_L, z_H + input_embeddings)
            z_H = L_level(z_H, z_L)
        for _L_step in range(L_cycles):  # with grad
            z_L = L_level(z_L, z_H + input_embeddings)
        z_H = L_level(z_H, z_L)
        """
        trajectory = {'z_states': [], 'y_states': []} if return_trajectory else None
        
        # H_cycles-1 without gradients (where H_cycles = n_refinement_steps)
        if self.n_refinement_steps > 1:
            with torch.no_grad():
                for _H_step in range(self.n_refinement_steps - 1):
                    # Inner loop: L_cycles (reasoning steps) - update z_L using (z_L, z_H + x)
                    for _L_step in range(self.n_reasoning_steps):
                        _, _, z_new = self.forward_pass(x, y, z, mask)
                        z = z_new
                        if return_trajectory:
                            trajectory['z_states'].append(z.detach().clone())
                    
                    # After inner loop: update z_H using (z_H, z_L)
                    _, y_new, _ = self.forward_pass(x, y, z, mask)
                    y = y_new
                    if return_trajectory:
                        trajectory['y_states'].append(y.detach().clone())
        
        # 1 iteration with gradients
        for _L_step in range(self.n_reasoning_steps):
            _, _, z_new = self.forward_pass(x, y, z, mask)
            z = z_new
            if return_trajectory:
                trajectory['z_states'].append(z.detach().clone())
        
        _, y_new, _ = self.forward_pass(x, y, z, mask)
        y = y_new
        if return_trajectory:
            trajectory['y_states'].append(y.detach().clone())
        
        return (y, z, trajectory) if return_trajectory else (y, z)
    
    def forward(self, question_ids, answer_ids=None, latent_len=32, mask=None):
        batch_size = question_ids.size(0)
        device = question_ids.device
        
        x = self.embed_tokens(question_ids)
        
        if answer_ids is not None:
            y = self.embed_tokens(answer_ids)
        else:
            y = torch.randn(batch_size, latent_len, self.d_model, device=device) * 0.02
        
        z = torch.randn(batch_size, latent_len, self.d_model, device=device) * 0.02
        
        result = self.recursive_reasoning(x, y, z, mask)
        if isinstance(result, tuple):
            y_final, _ = result  # Unpack (y, z) or (y, z, trajectory)
        else:
            y_final = result
        
        logits = self.reverse_embedding(y_final)
        
        return logits
    
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
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def configure_optimizers(
        self,
        learning_rate=1e-4,
        warmup_steps=1000,
        total_steps=None,
        weight_decay=0.01,
        optimizer_type="adamw",
        scheduler_type="linear"
    ):
        """
        Configure optimizer and learning rate scheduler using HuggingFace utilities.
        
        This is a standard pattern used in PyTorch models to encapsulate
        optimizer and scheduler setup.
        
        Args:
            learning_rate: Base learning rate
            warmup_steps: Number of warmup steps for linear warmup
            total_steps: Total number of training steps (required for scheduler)
            weight_decay: Weight decay for optimizer
            optimizer_type: Type of optimizer ("adamw", "adam", "sgd")
            scheduler_type: Type of scheduler (e.g., "linear", "cosine", "cosine_with_restarts", 
                          "polynomial", "constant", "constant_with_warmup")
            
        Returns:
            optimizer: Configured optimizer
            scheduler: Learning rate scheduler (call scheduler.step() each training step)
        """
        # Setup optimizer
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
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}. Choose from: adamw, adam, sgd")
        
        # Setup scheduler using transformers.get_scheduler
        if total_steps is None:
            # If total_steps not provided, scheduler will need to be updated later
            scheduler = None
        else:
            scheduler = get_scheduler(
                name=scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        
        return optimizer, scheduler