import os
import sys
import logging
import json
from logging import StreamHandler, FileHandler, Formatter
from datetime import datetime
from pathlib import Path

class JSONFormatter(logging.Formatter):
    """Format logs as JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': f"{record.module}.{record.funcName}",
            'line': record.lineno,
        }
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_record, ensure_ascii=False)

class Logger:
    """Logger with console and file output support."""
    
    def __init__(
        self,
        name: str = 'transformer',
        log_dir: str | Path | None = None,
        log_file: str | None = None,
        log_level: int = logging.INFO,
        json_format: bool = False,
        console: bool = True,
        file: bool = True
    ) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Remove existing handlers
        
        # Setup formatters
        if json_format:
            formatter = JSONFormatter()
            console_formatter = formatter
        else:
            formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Console handler
        if console:
            handler = StreamHandler(sys.stdout)
            handler.setFormatter(console_formatter)
            self.logger.addHandler(handler)
        
        # File handler
        if file and log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / (log_file or f"{name}.log")
            
            handler = FileHandler(log_path, encoding='utf-8')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs) -> None:
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs) -> None:
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs) -> None:
        self.logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs) -> None:
        self.logger.exception(msg, *args, **kwargs)
    
    def log_metrics(
        self,
        metrics: dict,
        step: int | None = None,
        prefix: str = ''
    ) -> None:
        """Log metrics in a readable format."""
        prefix = f"{prefix}_" if prefix and not prefix.endswith('_') else prefix or ''
        log_msg = [f"Step {step}:"] if step is not None else []
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                log_msg.append(f"{prefix}{key}: {value:.4f}")
            else:
                log_msg.append(f"{prefix}{key}: {value}")
        
        self.info(' '.join(log_msg))

def setup_logging(
    name: str = 'transformer',
    log_dir: str | Path | None = None,
    log_level: int | str = logging.INFO,
    json_format: bool = False
) -> Logger:
    """Configure and return a logger instance."""
    level = getattr(logging, log_level.upper()) if isinstance(log_level, str) else log_level
    return Logger(
        name=name,
        log_dir=log_dir,
        log_level=level,
        json_format=json_format,
        console=True,
        file=log_dir is not None
    )

def setup_logging(log_dir=None, name='transformer', level=logging.INFO, json_format=False):
    """
    Set up logging with the specified configuration.
    
    Args:
        log_dir: Directory to save log files. If None, no file logging.
        name: Logger name.
        level: Logging level.
        json_format: Whether to use JSON format for logs.
        
    Returns:
        Configured Logger instance.
    """
    logger = Logger(
        name=name,
        log_dir=log_dir,
        log_level=level,
        json_format=json_format,
        console=True,
        file=log_dir is not None
    )
    return logger

# Default logger instance
default_logger = setup_logging(log_dir=None)
