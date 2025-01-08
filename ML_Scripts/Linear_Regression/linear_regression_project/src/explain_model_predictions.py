# FILE: /linear_regression_project/src/explain_model_predictions.py

import shap
from get_logger import get_logger
from config import config

def explain_model_predictions(model, test_df):
    print("Executing explain_model_predictions.py")
    logger = get_logger('explain_model_predictions')
    logger.info("Explaining model predictions using SHAP...")

    X_test = test_df.drop(columns=[config['target_column']])
    y_test = test_df[config['target_column']]

    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")

    logger.info("SHAP explanations generated successfully.")