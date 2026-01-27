import os
import json
import shutil

import torch
import torch.nn as nn
from accelerate import Accelerator
from loguru import logger

from src.modeling.trm import TRM
from src.utils.logger import setup_logger, create_output_dir
from src.dataset.puzzle_dataset import create_dataloader


def train(
    config_path=None,
    vocab_size=None,
    d_model=None,
    n_heads=None,
    d_ff=None,
    n_layers=None,
    n_reasoning_steps=None,
    n_refinement_steps=None,
    use_attention=None,
    max_seq_len=None,
    dropout=None,
    batch_size=None,
    num_epochs=None,
    learning_rate=None,
    warmup_steps=None,
    max_norm=None,
    latent_len=None,
    device=None,
    data_paths=None,
    save_path=None,
    checkpoint_dir=None,
    eval_every_n_epochs=1,
    output_dir=None,
    seed=42
):
    setup_logger(output_dir=None, log_level="INFO")

    if output_dir is None:
        output_dir = create_output_dir()
    
    logger, _ = setup_logger(output_dir=output_dir, log_level="INFO")
    
    if config_path is None:
        config_path = 'configs/config.json'
    
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded config from {config_path}")
        shutil.copy(config_path, os.path.join(output_dir, "config.json"))
    else:
        logger.warning(f"Config file not found at {config_path}, using defaults")
    
    vocab_size = vocab_size if vocab_size is not None else config.get('vocab_size', 10000)
    d_model = d_model if d_model is not None else config.get('d_model', 256)
    n_heads = n_heads if n_heads is not None else config.get('n_heads', 4)
    d_ff = d_ff if d_ff is not None else config.get('d_ff', 1024)
    n_layers = n_layers if n_layers is not None else config.get('n_layers', 4)
    n_reasoning_steps = n_reasoning_steps if n_reasoning_steps is not None else config.get('n_reasoning_steps', 8)
    n_refinement_steps = n_refinement_steps if n_refinement_steps is not None else config.get('n_refinement_steps', 16)
    use_attention = use_attention if use_attention is not None else config.get('use_attention', True)
    max_seq_len = max_seq_len if max_seq_len is not None else config.get('max_seq_len', 512)
    dropout = dropout if dropout is not None else config.get('dropout', 0.1)
    latent_len = latent_len if latent_len is not None else config.get('latent_len', 32)
    
    batch_size = batch_size if batch_size is not None else config.get('batch_size', 32)
    num_epochs = num_epochs if num_epochs is not None else config.get('num_epochs', 50)
    learning_rate = learning_rate if learning_rate is not None else config.get('learning_rate', 1e-4)
    warmup_steps = warmup_steps if warmup_steps is not None else config.get('warmup_steps', 1000)
    max_norm = max_norm if max_norm is not None else config.get('max_norm', 1.0)
    weight_decay = config.get('weight_decay', 0.01)
    optimizer_type = config.get('optimizer_type', 'adamw')
    scheduler_type = config.get('scheduler_type', 'linear')
    seed = seed if seed is not None else config.get('seed', 42)
    save_checkpoint_every_n_steps = config.get('save_checkpoint_every_n_steps', None)
    
    if data_paths is None:
        data_paths = config.get('data_paths', [])
    
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    
    if not data_paths:
        logger.error("No data_paths specified in config or as parameter. Stopping Training.")
        exit(1)
    
    for data_path in data_paths:
        train_path = os.path.join(data_path, "train")
        if not os.path.exists(train_path):
            logger.error(f"Training data not found at {train_path}. Stopping Training.")
            exit(1)
    
    has_test_data = all(os.path.exists(os.path.join(data_path, "test")) for data_path in data_paths)
    if not has_test_data:
        logger.warning("Test data not found in one or more data paths. Evaluation will be skipped.")
    
    device = device if device is not None else config.get('device', None)
    
    if save_path is None:
        save_path = config.get('save_path', 'best_model.pt')
    if not os.path.isabs(save_path):
        save_path = os.path.join(output_dir, os.path.basename(save_path))
    
    if checkpoint_dir is None:
        checkpoint_dir = config.get('checkpoint_dir', None)
    if checkpoint_dir is not None and not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(output_dir, os.path.basename(checkpoint_dir))
    elif checkpoint_dir is None:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    mixed_precision = "bf16" if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else "no"
    
    accelerator = Accelerator(
        mixed_precision=mixed_precision
    )
    
    device = accelerator.device
    logger.info(f"Using device: {device}")
    if mixed_precision == "bf16":
        logger.info("BF16 mixed precision enabled via Accelerate")
    elif device.type == 'mps':
        logger.info("Using Apple Silicon GPU (MPS) with FP32 precision")
    else:
        logger.info("Using FP32 precision")
    
    rotary_base = config.get('rotary_base', 10000)
    
    model = TRM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        dropout=dropout,
        n_reasoning_steps=n_reasoning_steps,
        n_refinement_steps=n_refinement_steps,
        use_attention=use_attention,
        rotary_base=rotary_base
    )
    
    logger.info(f"Using Rotary Positional Embeddings (RoPE) with base={rotary_base}")
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"# Parameters: {n_params / 1e6:.2f}M")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    train_loader = None
    train_metadata = None
    val_loader = None
    val_metadata = None
    
    train_loader, train_metadata = create_dataloader(
        dataset_paths=data_paths,
        split="train",
        global_batch_size=batch_size,
        seed=seed,
        rank=0,
        world_size=1,
        test_set_mode=False,
        epochs_per_iter=1
    )
    logger.info(f"Training dataset: vocab_size={train_metadata.vocab_size}, seq_len={train_metadata.seq_len}")
    logger.info(f"Training on {train_metadata.total_puzzles} puzzles, {train_metadata.total_groups} groups")
    
    if vocab_size is None or vocab_size == config.get('vocab_size', 10000):
        vocab_size = train_metadata.vocab_size
        logger.info(f"Using vocab_size={vocab_size} from dataset metadata")
    
    val_loader = None
    val_metadata = None
    if has_test_data:
        val_loader, val_metadata = create_dataloader(
            dataset_paths=data_paths,
            split="test",
            global_batch_size=batch_size,
            seed=seed,
            rank=0,
            world_size=1,
            test_set_mode=True,
            epochs_per_iter=1
        )
        logger.info(f"Validation dataset: vocab_size={val_metadata.vocab_size}, seq_len={val_metadata.seq_len}")
        logger.info(f"Validation on {val_metadata.total_puzzles} puzzles")
    else:
        logger.info("Skipping validation dataloader creation (no test data available)")
    
    if train_loader is not None:
        if train_metadata:
            total_steps = train_metadata.total_groups * num_epochs
        else:
            total_steps = warmup_steps * 10
    else:
        total_steps = warmup_steps * 10
    
    optimizer, scheduler = model.configure_optimizers(
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        weight_decay=weight_decay,
        optimizer_type=optimizer_type,
        scheduler_type=scheduler_type
    )
    logger.info(f"Using optimizer: {optimizer_type}, scheduler: {scheduler_type}")
    
    # Prepare model and optimizer with Accelerate
    # Note: We don't prepare the dataloaders because PuzzleDataset is an IterableDataset
    # that yields tuples with strings, which Accelerate can't handle
    model, optimizer = accelerator.prepare(model, optimizer)
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    
    logger.info("="*50)
    logger.info("Starting training...")
    logger.info("="*50)
    
    best_loss = float('inf')
    best_val_accuracy = 0.0
    global_step = 0
    
    for epoch in range(num_epochs):
        if train_loader is None:
            logger.warning(f"Epoch {epoch+1}/{num_epochs} - No data, skipping")
            continue
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        batch_idx = 0
        for set_name, batch_dict, effective_batch_size in train_loader:
            question_ids = batch_dict["inputs"].to(device)
            answer_ids = batch_dict["labels"].to(device)
            
            optimizer.zero_grad()

            logits = model(question_ids, answer_ids, latent_len=latent_len)

            vocab_size = logits.size(-1)
            loss = criterion(
                logits.reshape(-1, vocab_size),
                answer_ids.reshape(-1)
            )
            accelerator.backward(loss)
            
            if max_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            current_lr = optimizer.param_groups[0]['lr']
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = epoch_loss / num_batches
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, "
                      f"Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
            
            # Save checkpoint every N steps if configured
            if save_checkpoint_every_n_steps and global_step % save_checkpoint_every_n_steps == 0:
                checkpoint_path = f'{checkpoint_dir}/checkpoint_step_{global_step}.pt'
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss / num_batches,
                    'val_loss': None,
                    'val_accuracy': None,
                }, checkpoint_path)
                logger.info(f"✓ Saved step checkpoint: {checkpoint_path}")
            
            batch_idx += 1
        
        avg_loss = epoch_loss / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}, "
              f"LR: {current_lr:.2e}, Steps: {global_step}")
        
        val_loss = None
        val_accuracy = None
        if val_loader is not None and (epoch + 1) % eval_every_n_epochs == 0:
            val_loss, val_accuracy = evaluate_accuracy(
                accelerator.unwrap_model(model), 
                val_loader, 
                device, 
                criterion,
                latent_len
            )
            logger.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
        
        loss_to_compare = val_loss if val_loss is not None else avg_loss
        if loss_to_compare < best_loss:
            best_loss = loss_to_compare
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'global_step': global_step,
            }, save_path)
            logger.info(f"✓ Saved new best model! (Loss: {best_loss:.4f})")
        
        # Save epoch-based checkpoint (every 10 epochs by default, or if save_checkpoint_every_n_steps not set)
        if checkpoint_dir is not None:
            save_epoch_checkpoint = False
            if save_checkpoint_every_n_steps is None:
                # If step-based checkpointing not configured, use epoch-based (every 10 epochs)
                save_epoch_checkpoint = (epoch + 1) % 10 == 0
            elif (epoch + 1) % 10 == 0:
                # Also save epoch checkpoints every 10 epochs as backup
                save_epoch_checkpoint = True
            
            if save_epoch_checkpoint:
                checkpoint_path = f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt'
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                }, checkpoint_path)
                logger.info(f"✓ Saved epoch checkpoint: {checkpoint_path}")
    
    logger.info("="*50)
    logger.info("Training complete!")
    logger.info(f"Best loss: {best_loss:.4f}")
    if best_val_accuracy > 0:
        logger.info(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    logger.info("="*50)
    
    # Save final best model if it hasn't been saved recently
    unwrapped_model = accelerator.unwrap_model(model)
    final_model_path = os.path.join(output_dir, "final_best_model.pt")
    torch.save({
        'epoch': num_epochs - 1,
        'global_step': global_step,
        'model_state_dict': unwrapped_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': best_loss,
        'val_loss': None,
        'val_accuracy': best_val_accuracy if best_val_accuracy > 0 else None,
    }, final_model_path)
    logger.info(f"✓ Saved final best model: {final_model_path}")
    
    logger.info(f"All outputs saved to: {output_dir}")


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
    """
    Visualize how the model thinks.
    """
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
    train(config_path='configs/config.json')
