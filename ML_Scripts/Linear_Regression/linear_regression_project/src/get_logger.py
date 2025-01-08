# FILE: /linear_regression_project/src/get_logger.py

import logging
from logging.handlers import RotatingFileHandler
import os
from config import config

def get_logger(name):
    print("Executing get_logger.py")
    log_path = config['log_path']
    handler = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=2)
    console_handler = logging.StreamHandler()
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        logger.addHandler(handler)
        logger.addHandler(console_handler)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    return logger