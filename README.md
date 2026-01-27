# TRM: Transformer Reasoning Model

Less is More: Recursive Reasoning with Tiny Networks

## Installation

### Requirements

```bash
pip install torch torchvision torchaudio
pip install accelerate transformers numpy loguru
```

### Mac-Specific Notes

**Apple Silicon (M1/M2/M3 Macs):**
- PyTorch will automatically use MPS (Metal Performance Shaders) for GPU acceleration
- No CUDA support (CUDA is NVIDIA-only)
- Mixed precision (bf16) is not available, will use FP32
- Training will be faster than CPU but slower than CUDA GPUs

**Intel Macs:**
- Will use CPU only
- Training will be slower but still functional

## Usage

### Basic Training

```python
from main import train

model = train(
    config_path='configs/config.json',
    train_questions=your_questions,
    train_answers=your_answers,
    tokenizer=your_tokenizer
)
```

### Sudoku Example

```python
python main.py
```

This will train a Sudoku solver using dummy data as an example.

### Custom Dataset

See `main.py` for examples of how to:
- Create custom tokenizers
- Format your data
- Train with validation
- Evaluate models

## Configuration

Edit `configs/config.json` to customize:
- Model architecture (d_model, n_layers, etc.)
- Training hyperparameters (batch_size, learning_rate, etc.)
- Optimizer and scheduler types

## Output Directory

Training runs automatically create a timestamped output directory (e.g., `outputs/run_20250126_143022/`) containing:
- `training.log` - Complete training log with all messages
- `config.json` - Copy of the configuration used
- `best_model.pt` - Best model checkpoint
- `checkpoints/` - Periodic checkpoints (every 10 epochs)

## Logging

The project uses [loguru](https://github.com/Delgan/loguru) for unified logging. All logs are:
- Displayed in the console with colors
- Saved to `training.log` in the output directory
- Automatically rotated when files exceed 100MB
- Retained for 10 days with compression

## Device Detection

The code automatically detects:
- **CUDA** (NVIDIA GPUs on Linux/Windows)
- **MPS** (Apple Silicon Macs)
- **CPU** (fallback)

You can override by setting `device` parameter in the config or function call.
