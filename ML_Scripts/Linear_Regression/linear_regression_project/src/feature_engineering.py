# FILE: /linear_regression_project/src/feature_engineering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, LabelEncoder
from feature_selection import feature_selection
from get_logger import get_logger
from config import config

def feature_engineering(df, config):
    print("Executing feature_engineering.py")
    logger = get_logger('feature_engineering')
    logger.info("--- Feature Engineering ---")

    # Encode categorical variables
    logger.info("Encoding categorical variables...")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    logger.info("Categorical variables encoded.")

    # Feature Engineering for Normalization
    logger.info("Performing feature engineering for normalization...")
    scaler_name = config['feature_engineering']['scaler']
    if scaler_name:
        scaler = eval(scaler_name)()
        numeric_cols = df.select_dtypes(include=[pd.np.number]).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        logger.info("Normalization completed.")

    # Perform feature selection if specified in the config
    if config['feature_engineering'].get('perform_feature_selection', False):
        method = config['feature_selection']['method']
        threshold = config['feature_selection'].get('threshold', 0.1)
        df = feature_selection(df, config['target_column'], method=method, threshold=threshold)
        logger.info("Feature selection completed.")

    # Generate synthetic features if specified in the config
    if config['feature_engineering'].get('generate_synthetic_features', False):
        df = generate_synthetic_features(df)
        logger.info("Synthetic features generated.")

    logger.info("Feature Engineering Summary:\nApplied %s for normalization.", scaler_name)
    return df