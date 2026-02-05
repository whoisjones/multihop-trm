import os
import logging
import wandb

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from accelerate import Accelerator

from src.modeling.trm import TRM
from src.utils import setup_logger, PRECISION_MAP
from src.utils.dataset import GroupBatchSampler, group_collate_fn, simple_collate_fn, infinite_dataloader


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def run(cfg: DictConfig):
    output_dir = HydraConfig.get().run.dir
    os.makedirs(output_dir, exist_ok=True)
    wandb.init(project=cfg.project_name, name=output_dir.split("/")[-1], config=OmegaConf.to_container(cfg, resolve=True))

    logger, _ = setup_logger(output_dir=output_dir, log_level=cfg.log_level)
    logger.info(f"Output directory: {output_dir}")

    config_save_path = os.path.join(output_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    dataset = load_dataset(cfg.data.data_paths)
    
    train_sampler = GroupBatchSampler(
        dataset=dataset["train"],
        batch_size=cfg.training.batch_size,
        group_column="group",
        shuffle=True,
        drop_last=cfg.training.get("drop_last", False),
    )
    
    train_dataloader = DataLoader(
        dataset["train"],
        batch_sampler=train_sampler,
        collate_fn=group_collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        dataset["test"],
        batch_size=cfg.training.batch_size,
        collate_fn=simple_collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    if cfg.training.mixed_precision == "bf16":
        mixed_precision = "bf16" if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else "no"
    elif cfg.training.mixed_precision == "fp16":
        mixed_precision = "fp16" if (device.type == 'cuda') else "no"
    else:
        mixed_precision = cfg.training.mixed_precision

    if device.type == "mps":
        mixed_precision = "no"
    
    logger.info(f"Device: {device.type}, mixed precision: {mixed_precision}")

    accelerator = Accelerator(
        mixed_precision=mixed_precision
    )
    
    model = TRM(
        backbone=cfg.model.backbone,
        vocab_size=cfg.data.vocab_size,
        puzzle_len=cfg.data.puzzle_len,
        deep_supervision_steps=cfg.model.deep_supervision_steps,
        deep_supervision_exploration_prob=cfg.model.deep_supervision_exploration_prob,
        d_hidden=cfg.model.d_hidden,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        n_reasoning_steps=cfg.model.n_reasoning_steps,
        n_latent_steps=cfg.model.n_latent_steps,
        tie_embeddings=cfg.model.tie_embeddings,
        dropout=cfg.model.dropout,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"# Parameters: {n_params / 1e6:.2f}M")
    
    optimizer, scheduler = model.configure_optimizers(
        optimizer_type=cfg.training.optimizer_type,
        scheduler_type=cfg.training.scheduler_type,
        learning_rate=cfg.training.learning_rate,
        warmup_steps=cfg.training.warmup_steps,
        total_steps=cfg.training.num_steps,
        weight_decay=cfg.training.weight_decay
    )
    logger.info(f"Using optimizer: {cfg.training.optimizer_type}, scheduler: {cfg.training.scheduler_type}")
    
    model, train_dataloader, val_dataloader, scheduler, optimizer = accelerator.prepare(
        model, train_dataloader, val_dataloader, scheduler, optimizer
    )
    
    train_iter = infinite_dataloader(train_dataloader)
    
    logger.info("="*50)
    logger.info("Starting training...")
    logger.info("="*50)
    
    best_loss = float('inf')
    best_val_accuracy = 0.0
    global_step = 0
    
    while global_step < cfg.training.num_steps:
        model.train()
        batch = next(train_iter)

        if global_step == 0:
            dtype = PRECISION_MAP.get(mixed_precision, torch.float32)
            carry = {}

            carry['answer'] = torch.zeros(
                cfg.training.batch_size,
                cfg.data.puzzle_len,
                cfg.model.d_hidden,
                dtype=dtype,
                device=device
            )
            carry['latent'] = torch.zeros(
                cfg.training.batch_size,
                cfg.data.puzzle_len,
                cfg.model.d_hidden,
                dtype=dtype,
                device=device
            )
            carry['steps'] = torch.zeros((cfg.training.batch_size,), dtype=torch.int32, device=device)
            carry['halted'] = torch.ones((cfg.training.batch_size,), dtype=torch.bool, device=device)
            carry['current_data'] = {k: torch.zeros_like(v, device=device) for k, v in batch.items()}

        optimizer.zero_grad()

        outputs, carry = model(batch, carry)
        labels = carry['current_data']['labels']

        with torch.no_grad():
            outputs['predictions'] = outputs['logits'].argmax(dim=-1)
            mask = (labels != -100)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
            cell_tp = mask & (outputs['predictions'] == labels)
            seq_tp = cell_tp.all(dim=-1)
            valid_metrics = carry['halted'] & (loss_counts > 0)

        seq_loss = ((F.cross_entropy(outputs['logits'].transpose(1, 2), labels, reduction='none') * mask) / loss_divisor).sum() / cfg.training.batch_size
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs['q_halt_logits'], seq_tp.to(outputs['q_halt_logits'].dtype), reduction='sum') / cfg.training.batch_size
        loss = (cfg.model.seq_loss_weight * seq_loss + cfg.model.q_halt_loss_weight * q_halt_loss) 

        metrics = {
            "valid_sequences": valid_metrics.sum(),
            "cell_accuracy": torch.where(valid_metrics, (cell_tp / loss_divisor).sum(-1), 0).sum(),
            "game_accuracy": (valid_metrics & seq_tp).sum(),
            "q_halt_correct": (valid_metrics & ((outputs['q_halt_logits'] >= 0) == seq_tp)).sum(),
            "steps": torch.where(valid_metrics, carry['steps'], 0).sum(),
            "seq_loss": seq_loss.detach(),
            "q_halt_loss": q_halt_loss.detach()
        }

        accelerator.backward(loss)

        if cfg.training.max_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), max_norm=cfg.training.max_norm)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        global_step += 1

        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k].float() for k in metric_keys])
        metric_values = accelerator.reduce(metric_values, reduction="sum")
        
        if accelerator.is_main_process:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            count = max(reduced_metrics["valid_sequences"], 1)  # Avoid NaNs
            reduced_metrics = {
                f"train/{k}": v / count if not k.endswith("loss") else v 
                for k, v in reduced_metrics.items()
            }
            reduced_metrics["train/lr"] = optimizer.param_groups[0]['lr']

        wandb.log(reduced_metrics, step=global_step)

        if global_step % cfg.training.log_interval == 0:
            logger.info(f"Step {global_step}, Loss: {loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

        # if global_step % cfg.training.eval_interval == 0:
        #     model.eval()
        #     metrics = {}
        #     with torch.no_grad():
        #         for batch in val_dataloader:
        #             labels = batch["labels"]
        #             logits, (y, z), (q_halt_logits, q_continue_logits) = model(batch, steps, halted, current_data)
        #             predictions = logits.argmax(dim=-1)
                    
        #             metrics['accuracy'] = (predictions == labels).float().mean().item()
            
        #     if metrics['accuracy'] > best_val_accuracy:
        #         pass # save

        #     with torch.no_grad():
        #         # Step
        #         new_steps = new_steps + 1
        #         is_last_step = new_steps >= self.config.halt_max_steps
                
        #         halted = is_last_step

        #         if self.training and (self.config.halt_max_steps > 1):
        #             halted = halted | (q_halt_logits > 0)
        #             min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
        #             halted = halted & (new_steps >= min_halt_steps)

    wandb.finish()


@torch.no_grad()
def evaluate_accuracy(model, test_loader, device, criterion, latent_len=32):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0
    
    for set_name, batch_dict, effective_batch_size in test_loader:
        questions = batch_dict["inputs"].to(device)
        answers = batch_dict["labels"].to(device)
        
        logits = model(questions, answers, latent_len=latent_len)
        predictions = logits.argmax(dim=-1)
        
        vocab_size = logits.size(-1)
        loss = criterion(
            logits.reshape(-1, vocab_size),
            answers.reshape(-1)
        )
        total_loss += loss.item()
        num_batches += 1
        
        mask = answers != 0
        correct += (predictions[mask] == answers[mask]).sum().item()
        total += mask.sum().item()
    
    accuracy = (correct / total * 100) if total > 0 else 0.0
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, accuracy


@torch.no_grad()
def generate_answer(model, question_text, tokenizer, device, max_length=50, latent_len=32):
    logger, _ = setup_logger()
    
    model.eval()
    
    question_tokens = tokenizer.encode(question_text)
    question_ids = torch.tensor([question_tokens]).to(device)
    
    logger.info(f"\nQuestion: {question_text}")
    logger.info("Thinking...")
    
    generated_ids = model.generate(
        question_ids,
        max_length=max_length,
        latent_len=latent_len,
        temperature=0.7
    )
    
    answer_tokens = generated_ids[0].cpu().tolist()
    answer_text = tokenizer.decode(answer_tokens)
    
    logger.info(f"Answer: {answer_text}\n")
    return answer_text


@torch.no_grad()
def visualize_reasoning_process(model, question_ids, answer_ids, device, latent_len=32):
    logger, _ = setup_logger()
    
    model.eval()
    
    if not isinstance(question_ids, torch.Tensor) or question_ids.device != device:
        question_ids = question_ids.to(device)
    if not isinstance(answer_ids, torch.Tensor) or answer_ids.device != device:
        answer_ids = answer_ids.to(device)
    
    x = model.embed_tokens(question_ids)
    y = model.embed_tokens(answer_ids)
    z = torch.randn(question_ids.size(0), latent_len, model.d_model, device=device) * 0.02
    
    result = model.recursive_reasoning(
        x, y, z, return_trajectory=True
    )
    y_final, z_final, trajectory = result
    
    logger.info("\n Reasoning Evolution:")
    logger.info("=" * 50)
    
    # Show how z evolves (reasoning)
    logger.info("\n Reasoning Stream (z):")
    for i, z_state in enumerate(trajectory['z_states'][:5]):
        z_norm = z_state.norm(dim=-1).mean().item()
        logger.info(f"  Step {i+1}: norm = {z_norm:.4f}")
    
    # Show how y evolves (answer)
    logger.info("\n Answer Stream (y):")
    for i, y_state in enumerate(trajectory['y_states'][:5]):
        y_norm = y_state.norm(dim=-1).mean().item()
        logger.info(f"  Step {i+1}: norm = {y_norm:.4f}")
    
    logger.info("\n Final answer generated!")


if __name__ == "__main__":
    run()
