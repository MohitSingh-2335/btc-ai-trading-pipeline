import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name="crypto_bot", log_file="logs/app.log"):
    """
    Sets up a logger that writes to a file (with rotation) and the console.
    """
    # 1. Create Logs Directory
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 2. Configure Logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate logs if function is called twice
    if logger.hasHandlers():
        return logger

    # 3. File Handler (Writes to file, max 5MB, keeps 3 backups)
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 4. Console Handler (Prints to screen)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(message)s') # Simpler format for screen
    console_handler.setFormatter(console_formatter)

    # 5. Add Handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
