import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Initializes a multi-GPU-friendly python command line logger that logs on all processes
        with their rank prefixed in the log message.

        :param name: The name of the logger. Default is ``__name__``.
        :param rank_zero_only: Whether to force all logs to only occur on the rank zero process. Default is `False`.
        :param extra: (Optional) A dict-like object which provides contextual information. See `logging.LoggerAdapter`.
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        """Delegate a log call to the underlying logger, after prefixing its message with the rank
        of the process it's being logged from. If `'rank'` is provided, then the log will only
        occur on that rank/process.

        :param level: The level to log at. Look at `logging.__init__.py` for more information.
        :param msg: The message to log.
        :param rank: The rank to log at.
        :param args: Additional args to pass to the underlying logging function.
        :param kwargs: Any additional keyword args to pass to the underlying logging function.
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None:
                    self.logger.log(level, msg, *args, **kwargs)
                elif current_rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)


def setup_logger(output_dir: str = None, log_level: str = "INFO"):
    """
    Setup RankedLogger with file and console handlers.
    Only logs on rank 0 when running with distributed training (torchrun).
    
    Args:
        output_dir: Directory to save log file. If None, logs only to console.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        logger: Configured RankedLogger instance
        output_dir: Path to output directory (created if needed)
    """
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    
    # Set up rank for distributed training
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank_zero_only.rank = dist.get_rank()
        else:
            # Not in distributed mode, set rank to 0
            rank_zero_only.rank = 0
    except (ImportError, AttributeError):
        # PyTorch not available or distributed not initialized
        rank_zero_only.rank = 0
    
    # Create RankedLogger with rank_zero_only=True to only log on rank 0
    logger = RankedLogger(name="multihop_trm", rank_zero_only=True)
    
    # Get the underlying logging.Logger to configure handlers
    base_logger = logger.logger
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    base_logger.setLevel(level)
    
    # Prevent propagation to root logger to avoid duplicates
    base_logger.propagate = False
    
    # Remove existing handlers to avoid duplicates
    base_logger.handlers.clear()
    
    # Create formatter
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Add console handler (only on rank 0)
    if rank_zero_only.rank == 0:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        base_logger.addHandler(console_handler)
    
    # Create output directory if provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Add file handler (only on rank 0)
        if rank_zero_only.rank == 0:
            log_file = output_path / "training.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=100 * 1024 * 1024,  # 100 MB
                backupCount=10
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(file_formatter)
            base_logger.addHandler(file_handler)
            
            logger.info(f"Logging to: {log_file}")
            return logger, str(output_path)
    
    return logger, None