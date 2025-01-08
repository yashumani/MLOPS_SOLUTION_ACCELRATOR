# FILE: /linear_regression_project/src/visualize_model_performance.py

import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.regression import predict_model as reg_predict_model
from get_logger import get_logger
from config import config

def visualize_model_performance(best_trial, test_df):
    print("Executing visualize_model_performance.py")
    logger = get_logger('visualize_model_performance')
    logger.info("Visualizing Model Performance")

    y_true = test_df[config['target_column']]

    if isinstance(best_trial, dict) and 'model' in best_trial.params:
        model_name = best_trial.params['model']
        if model_name == 'ridge':
            model = Ridge(alpha=best_trial.params['alpha'])
            model.fit(test_df.drop(columns=[config['target_column']]), y_true)
            y_pred = model.predict(test_df.drop(columns=[config['target_column']]))
        else:
            automl = AutoML()
            automl_settings = {
                "time_budget": config['flaml']['time_budget'],
                "metric": config['flaml']['metric'],
                "task": config['flaml']['task'],
                "log_file_name": 'flaml.log',
                "verbose": 0
            }
            automl.fit(test_df.drop(columns=[config['target_column']]), y_true, **automl_settings)
            y_pred = automl.predict(test_df.drop(columns=[config['target_column']]))

        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Model: True vs Predicted Values')
        plt.savefig(os.path.join(config['reports_path'], "true_vs_predicted.png"))
        plt.close()

        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residuals')
        plt.title('Residuals Distribution')
        plt.savefig(os.path.join(config['reports_path'], "residuals_distribution.png"))
        plt.close()