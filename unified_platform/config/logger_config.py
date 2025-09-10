"""
Unicode-Safe Logger Configuration
=================================

Configures logging to properly handle Unicode characters and emojis
on Windows systems, fixing the cp1252 encoding issues.
"""

import logging
import sys
import os
from typing import Optional


class UnicodeFileHandler(logging.FileHandler):
    """File handler that properly handles Unicode characters."""
    
    def __init__(self, filename, mode='a', encoding='utf-8', delay=False):
        super().__init__(filename, mode, encoding, delay)


class UnicodeStreamHandler(logging.StreamHandler):
    """Stream handler that properly handles Unicode characters."""
    
    def __init__(self, stream=None):
        super().__init__(stream)
        
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            
            # Try to write with UTF-8 encoding
            if hasattr(stream, 'buffer'):
                # For stdout/stderr, write to buffer with UTF-8
                stream.buffer.write((msg + self.terminator).encode('utf-8'))
                stream.buffer.flush()
            else:
                # Fallback to regular write
                stream.write(msg + self.terminator)
                if hasattr(stream, 'flush'):
                    stream.flush()
        except Exception:
            # If Unicode fails, replace problematic characters
            try:
                msg_safe = msg.encode('ascii', 'replace').decode('ascii')
                self.stream.write(msg_safe + self.terminator)
                if hasattr(self.stream, 'flush'):
                    self.stream.flush()
            except Exception:
                self.handleError(record)


def setup_unicode_logger(name: str, 
                        log_file: Optional[str] = None,
                        log_level: int = logging.INFO,
                        console_output: bool = True) -> logging.Logger:
    """
    Setup a logger that properly handles Unicode characters and emojis.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        log_level: Logging level
        console_output: Whether to output to console
    
    Returns:
        Configured logger
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter with Unicode support
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler if specified
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = UnicodeFileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if specified
    if console_output:
        console_handler = UnicodeStreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def configure_genesis_logging():
    """
    Configure Genesis logging to handle Unicode properly.
    This should be called before Genesis initialization.
    """
    
    # Set environment variables for better Unicode support
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # Configure Python's default encoding
    if hasattr(sys, 'set_int_max_str_digits'):
        # Python 3.11+ - ensure large numbers don't cause issues
        sys.set_int_max_str_digits(10000)
    
    # Configure logging root handler to use UTF-8
    root_logger = logging.getLogger()
    
    # Replace any existing handlers with Unicode-safe ones
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            root_logger.removeHandler(handler)
            # Add Unicode-safe handler
            unicode_handler = UnicodeStreamHandler(handler.stream)
            unicode_handler.setFormatter(handler.formatter)
            unicode_handler.setLevel(handler.level)
            root_logger.addHandler(unicode_handler)


def create_simulation_logger(log_dir: str = "simulation_logs") -> logging.Logger:
    """
    Create a logger specifically for simulation with emoji support.
    
    Args:
        log_dir: Directory for log files
    
    Returns:
        Configured simulation logger
    """
    
    # Configure Genesis logging first
    configure_genesis_logging()
    
    # Create simulation log file path
    log_file = os.path.join(log_dir, "simulation_unicode.log")
    
    # Setup logger with Unicode support
    logger = setup_unicode_logger(
        name="simulation",
        log_file=log_file,
        log_level=logging.INFO,
        console_output=True
    )
    
    # Test the logger with emojis
    logger.info("🚀 Unicode logger initialized successfully!")
    logger.info("✅ Emoji support enabled")
    logger.info("🎯 Ready for simulation logging")
    
    return logger


# Emoji constants for consistent usage
class Emojis:
    """Emoji constants for logging."""
    
    SUCCESS = "✅"
    ERROR = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"
    ROCKET = "🚀"
    TARGET = "🎯"
    ROBOT = "🤖"
    GEAR = "⚙️"
    CHART = "📊"
    FILE = "📁"
    CHECKMARK = "✓"
    CROSS = "✗"
    ARROW_RIGHT = "→"
    ARROW_DOWN = "↓"
    TOOLS = "🔧"
    LIGHTNING = "⚡"
    FIRE = "🔥"
    STAR = "⭐"
    BRAIN = "🧠"
    EYE = "👁️"
    HAND = "✋"


if __name__ == "__main__":
    # Test the Unicode logger
    print("Testing Unicode Logger...")
    
    logger = create_simulation_logger()
    
    # Test various emojis
    logger.info(f"{Emojis.ROCKET} Starting test...")
    logger.info(f"{Emojis.ROBOT} Robot initialized")
    logger.info(f"{Emojis.GEAR} Configuration loaded")
    logger.info(f"{Emojis.TARGET} Target set")
    logger.info(f"{Emojis.SUCCESS} Test completed successfully!")
    
    print("✅ Unicode logger test completed!")
