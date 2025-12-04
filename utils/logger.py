#!/usr/bin/env python3
"""
Structured logging utility for FITTR RAG Chatbot.

Provides production-grade logging with:
- JSON structured output for production
- Human-readable format for development
- Rotating file handlers
- Multiple log levels
- Context enrichment
"""

import logging
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any
from functools import wraps
import time
import traceback


class JsonFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    
    Outputs logs in JSON format for easy parsing by log aggregators
    like CloudWatch, Datadog, or ELK stack.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': ''.join(traceback.format_exception(*record.exc_info))
            }
        
        # Add extra context from record
        if hasattr(record, 'extra_data'):
            log_data['context'] = record.extra_data
        
        return json.dumps(log_data, default=str)


class ContextFilter(logging.Filter):
    """
    Filter to add contextual information to log records.
    
    Adds app-level context like environment, version, etc.
    """
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to record."""
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


def setup_logger(
    name: str,
    level: str = "INFO",
    log_dir: str = "logs",
    json_format: bool = False,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up structured logger with console and file handlers.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        json_format: If True, use JSON formatting for files
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup log files to keep
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Application started", extra={
        ...     'extra_data': {'user_id': '123', 'query': 'test'}
        ... })
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # --- Console Handler (Human-readable) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    console_format = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # --- File Handler (Rotating) ---
    log_file = os.path.join(log_dir, f'{name.replace(".", "_")}.log')
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # Capture all levels in file
    
    # Use JSON format for production, human-readable for development
    if json_format or os.getenv('ENVIRONMENT', 'development') == 'production':
        file_handler.setFormatter(JsonFormatter())
    else:
        file_format = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
    
    # --- Error File Handler (Errors and Critical only) ---
    error_log_file = os.path.join(log_dir, 'errors.log')
    error_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JsonFormatter())
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    
    # Add context filter with environment info
    context = {
        'environment': os.getenv('ENVIRONMENT', 'development'),
        'app_name': 'fittr-rag-chatbot'
    }
    logger.addFilter(ContextFilter(context))
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger or create new one with default config.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger doesn't have handlers, set it up
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


def log_execution_time(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger instance (if None, creates one)
    
    Example:
        >>> @log_execution_time()
        ... def process_query(query: str):
        ...     # Processing logic
        ...     return result
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            start_time = time.time()
            function_name = func.__name__
            
            logger.debug(f"Starting {function_name}", extra={
                'extra_data': {
                    'function': function_name,
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                }
            })
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(f"{function_name} completed successfully", extra={
                    'extra_data': {
                        'function': function_name,
                        'duration_seconds': round(duration, 3),
                        'status': 'success'
                    }
                })
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                logger.error(f"{function_name} failed", extra={
                    'extra_data': {
                        'function': function_name,
                        'duration_seconds': round(duration, 3),
                        'status': 'error',
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    }
                }, exc_info=True)
                
                raise
        
        return wrapper
    return decorator


def log_rag_query(logger: logging.Logger, query: str, context: Dict[str, Any]):
    """
    Log RAG query with structured context.
    
    Args:
        logger: Logger instance
        query: User query
        context: Additional context (route, duration, etc.)
    
    Example:
        >>> log_rag_query(logger, "How to lose weight?", {
        ...     'route': 'knowledge',
        ...     'duration': 2.3,
        ...     'num_sources': 3
        ... })
    """
    logger.info("RAG query processed", extra={
        'extra_data': {
            'query': query[:100],  # Truncate long queries
            'query_length': len(query),
            **context
        }
    })


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: Dict[str, Any],
    message: str = "An error occurred"
):
    """
    Log error with rich contextual information.
    
    Args:
        logger: Logger instance
        error: Exception object
        context: Additional context about the error
        message: Custom error message
    
    Example:
        >>> try:
        ...     # Some operation
        ...     pass
        ... except Exception as e:
        ...     log_error_with_context(logger, e, {
        ...         'query': user_query,
        ...         'user_id': user_id
        ...     })
    """
    logger.error(message, extra={
        'extra_data': {
            'error_type': type(error).__name__,
            'error_message': str(error),
            **context
        }
    }, exc_info=True)


# Performance metrics logger
class PerformanceLogger:
    """
    Context manager for logging operation performance.
    
    Example:
        >>> logger = get_logger(__name__)
        >>> with PerformanceLogger(logger, "database_query"):
        ...     # Expensive operation
        ...     result = fetch_from_db()
    """
    
    def __init__(self, logger: logging.Logger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.operation}", extra={
            'extra_data': {'operation': self.operation, **self.context}
        })
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(f"{self.operation} completed", extra={
                'extra_data': {
                    'operation': self.operation,
                    'duration_seconds': round(duration, 3),
                    'status': 'success',
                    **self.context
                }
            })
        else:
            self.logger.error(f"{self.operation} failed", extra={
                'extra_data': {
                    'operation': self.operation,
                    'duration_seconds': round(duration, 3),
                    'status': 'error',
                    'error_type': exc_type.__name__,
                    'error_message': str(exc_val),
                    **self.context
                }
            }, exc_info=True)
        
        # Don't suppress exceptions
        return False


if __name__ == "__main__":
    # Demo usage
    logger = setup_logger(__name__, level="DEBUG")
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # With context
    logger.info("User query received", extra={
        'extra_data': {
            'user_id': '12345',
            'query': 'How to lose weight?',
            'ip_address': '192.168.1.1'
        }
    })
    
    # Performance logging
    with PerformanceLogger(logger, "test_operation", user_id="123"):
        time.sleep(0.1)
    
    print("\nLogs written to ./logs directory")
