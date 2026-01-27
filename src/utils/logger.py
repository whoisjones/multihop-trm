"""
Unified logging utility using loguru.
"""
import os
import sys
from datetime import datetime
from pathlib import Path
from loguru import logger


def setup_logger(output_dir: str = None, log_level: str = "INFO"):
    """
    Setup loguru logger with file and console handlers.
    
    Args:
        output_dir: Directory to save log file. If None, logs only to console.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        logger: Configured loguru logger instance
        output_dir: Path to output directory (created if needed)
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Create output directory if provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Add file handler
        log_file = output_path / "training.log"
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation="100 MB",
            retention="10 days",
            compression="zip"
        )
        
        logger.info(f"Logging to file: {log_file}")
        return logger, str(output_path)
    
    return logger, None


def create_output_dir(base_dir: str = "outputs") -> str:
    """
    Create a timestamped output directory.
    
    Args:
        base_dir: Base directory for outputs
    
    Returns:
        str: Path to created output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
