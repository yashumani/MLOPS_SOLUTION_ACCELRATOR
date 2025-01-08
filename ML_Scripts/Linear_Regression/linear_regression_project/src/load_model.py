import joblib
from get_logger import get_logger

def load_model(filename):
    logger = get_logger('load_model')
    logger.info(f"Loading model from {filename}...")
    model = joblib.load(filename)
    logger.info("Model loaded successfully.")
    return model