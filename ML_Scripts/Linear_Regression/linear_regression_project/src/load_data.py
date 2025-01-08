import pandas as pd
import numpy as np
from get_logger import get_logger
from config import config

def load_data(path):
    print("Executing load_data.py")
    logger = get_logger('load_data')
    logger.info("Loading dataset...")
    df = pd.read_csv(path)
    logger.info("Dataset loaded successfully.")
    print("Column types:\n", df.dtypes)
    print("Top 5 rows of the dataset:\n", df.head())
    
    # Check if the target column exists
    if config['target_column'] not in df.columns:
        logger.error(f"Target column '{config['target_column']}' not found in the dataset.")
        raise ValueError(f"Target column '{config['target_column']}' not found in the dataset.")
    
    # Check if there are enough numeric features for regression
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_features) < 2:  # At least one feature and the target column
        logger.error("Not enough numeric features for regression.")
        raise ValueError("Not enough numeric features for regression.")
    
    return df