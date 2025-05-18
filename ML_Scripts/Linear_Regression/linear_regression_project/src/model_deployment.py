import os
import joblib
from get_logger import get_logger
from config import config

def deploy_model(model):
    logger = get_logger('model_deployment')
    logger.info("Starting model deployment...")

    # Define the model deployment path
    model_deployment_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', config['model_deployment_path']))

    # Ensure the deployment directory exists
    os.makedirs(model_deployment_path, exist_ok=True)

    # Define the model file path
    model_file_path = os.path.join(model_deployment_path, 'model.joblib')

    # Save the model
    joblib.dump(model, model_file_path)
    logger.info(f"Model deployed successfully at: {model_file_path}")

    return model_file_path

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from clean_data import clean_data
    from feature_engineering import feature_engineering
    from model_selection import select_model

    # Load dataset
    df = pd.read_csv(config['data_path'])

    # Clean the dataset
    df = clean_data(df)

    # Perform feature engineering
    df = feature_engineering(df, config)

    # Split dataset into training and testing subsets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    X_train = train_df.drop(columns=[config['target_column']])
    y_train = train_df[config['target_column']]

    # Select the best model
    best_model = select_model(train_df, config)

    # Deploy the model
    model_file_path = deploy_model(best_model)
    print(f"Model deployed at: {model_file_path}")