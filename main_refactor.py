import wandb

import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset

from accelerate import Accelerator
import lightning as L
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

from src.logger import RankedLogger
from src.collate import collate_fn, infinite_dataloader
from src.ema import EMAHelper
from src.group_sampler import group_collate_fn, GroupBatchSampler

logger = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def run(cfg: DictConfig):
    if cfg.seed is not None:
        L.seed_everything(cfg.seed)

    wandb.init(project=cfg.project_name, config=OmegaConf.to_container(cfg, resolve=True))
    accelerator = Accelerator(mixed_precision=cfg.training.mixed_precision, log_with="wandb")
    dataset = load_dataset(cfg.data.data_paths)
    model = hydra.utils.instantiate(cfg.model, vocab_size=cfg.data.vocab_size, puzzle_len=cfg.data.puzzle_len)

    train_sampler = GroupBatchSampler(
        dataset=dataset["train"],
        batch_size=cfg.training.batch_size,
        group_column="group",
        shuffle=True,
        drop_last=True,
    )

    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=cfg.training.batch_size,
        collate_fn=group_collate_fn,
        pin_memory=True,
        sampler=train_sampler,
    )

    eval_dataloader = DataLoader(
        dataset["test"],
        batch_size=cfg.training.batch_size,
        pin_memory=True,
    )

    optimizer = hydra.utils.instantiate(cfg.training.optimizer, params=model.parameters())
    if cfg.training.puzzle_optimizer is not None:
        puzzle_optimizer = hydra.utils.instantiate(
            cfg.training.puzzle_optimizer, 
            params=model.embedding_layer.sparse_latent_puzzle_embedding.buffers(),
            world_size=accelerator.num_processes
        )
    else:
        puzzle_optimizer = None

    if cfg.training.scheduler is not None:
        scheduler = get_scheduler(
            name=cfg.training.scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.warmup_steps,
            num_training_steps=cfg.training.num_steps
        )
    else:
        scheduler = None

    if cfg.training.puzzle_optimizer is not None and cfg.training.puzzle_scheduler is not None:
        puzzle_scheduler = get_scheduler(
            name=cfg.training.puzzle_scheduler,
            optimizer=puzzle_optimizer,
            num_warmup_steps=cfg.training.warmup_steps,
            num_training_steps=cfg.training.num_steps
        )
    else:
        puzzle_scheduler = None

    model, train_dataloader, eval_dataloader, optimizer = accelerator.prepare(
        model, train_dataloader, eval_dataloader, optimizer
    )

    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)
    if puzzle_optimizer is not None:
        puzzle_optimizer = accelerator.prepare(puzzle_optimizer)
    if puzzle_scheduler is not None:
        puzzle_scheduler = accelerator.prepare(puzzle_scheduler)

    global_step = 0
    if cfg.training.ema:
        ema_helper = EMAHelper(mu=cfg.training.ema_mu)
        ema_helper.register(model)

    train_iter = infinite_dataloader(train_dataloader, expected_batch_size=cfg.training.batch_size)
    
    while global_step < cfg.training.num_steps:
        model.train()
        optimizer.zero_grad()
        batch = next(train_iter)

        if global_step == 0:
            carry = model.init_carry(batch)

        outputs, carry, metrics = model(batch, carry)

        accelerator.backward(outputs.loss)

        if cfg.training.max_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), max_norm=cfg.training.max_norm)

        optimizer.step()
        if puzzle_optimizer is not None:
            puzzle_optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if puzzle_scheduler is not None:
            puzzle_scheduler.step()
        
        global_step += 1

        if cfg.training.ema:
            ema_helper.update(model)

        if global_step % cfg.training.log_interval == 0:
            logger.info(f"Step {global_step}, Loss: {outputs.loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

        token_accuracies = torch.where(
            metrics.valid_metrics, 
            (metrics.token_correct / metrics.mask.sum(-1).unsqueeze(-1)).sum(-1), 
            0
        )
        seq_accuracies = (metrics.valid_metrics & metrics.seq_correct)
        q_halt_accuracies = (metrics.valid_metrics & ((outputs.q_halt_logits >= 0) == metrics.seq_correct))
        
        token_accuracies, seq_accuracies, q_halt_accuracies = accelerator.gather_for_metrics(
            (token_accuracies, seq_accuracies, q_halt_accuracies)
        )

        if accelerator.is_main_process:
            wandb.log({
                "train/steps": torch.where(metrics.valid_metrics, carry.steps, 0).cpu().numpy().sum() / cfg.training.batch_size,
                "train/seq_loss": outputs.seq_loss.cpu().numpy(),
                "train/q_halt_loss": outputs.q_halt_loss.cpu().numpy(),
                "train/token_accuracy": token_accuracies.cpu().numpy().mean(),
                "train/seq_accuracy": seq_accuracies.cpu().numpy().mean(),
                "train/q_halt_accuracy": q_halt_accuracies.cpu().numpy().mean(),
                "train/lr": optimizer.param_groups[0]['lr'],
            }, step=global_step)

    wandb.finish()

if __name__ == "__main__":
    run()
