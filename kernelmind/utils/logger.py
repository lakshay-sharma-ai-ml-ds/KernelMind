import logging
import sys
from pathlib import Path
from ..config import config

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    level = LOG_LEVELS.get(config.LOG_LEVEL, logging.INFO)
    logger.setLevel(level)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if config.LOG_FILE:
        log_dir = Path(config.LOG_FILE).parent
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(config.LOG_FILE)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def enable_debug():
    logging.getLogger("kernelmind").setLevel(logging.DEBUG)
    for handler in logging.getLogger("kernelmind").handlers:
        handler.setLevel(logging.DEBUG)

_logger = get_logger("kernelmind")
