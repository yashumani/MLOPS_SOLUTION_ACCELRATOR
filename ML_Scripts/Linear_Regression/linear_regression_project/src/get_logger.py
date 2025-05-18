# FILE: /ML_Scripts/Linear_Regression/linear_regression_project/src/get_logger.py

import logging
from logging.handlers import RotatingFileHandler
import os
from config import config

def get_logger(name):
    log_path = config['log_path']
    log_dir = os.path.dirname(log_path)

    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=2)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger