# FILE: /ML_Pipeline/src/model_deployment.py

import joblib
from get_logger import get_logger
from config import config

def deploy_model(model):
    logger = get_logger('model_deployment')
    logger.info("Deploying model...")

    model_path = os.path.join(config['model_path'], 'best_model.pkl')
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")