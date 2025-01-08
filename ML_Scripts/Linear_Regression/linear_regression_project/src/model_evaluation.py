# FILE: /ML_Pipeline/src/model_evaluation.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from get_logger import get_logger
from config import config

def evaluate_model(model, X_test, y_test):
    logger = get_logger('model_evaluation')
    logger.info("Evaluating model...")

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Mean Squared Error: {mse}")
    logger.info(f"R² Score: {r2}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.savefig(os.path.join(config['reports_path'], "true_vs_predicted.png"))
    plt.close()

    return mse, r2