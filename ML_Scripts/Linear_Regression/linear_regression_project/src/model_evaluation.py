from sklearn.metrics import mean_squared_error, r2_score
from get_logger import get_logger

def evaluate_model(model, X_test, y_test):
    logger = get_logger('model_evaluation')
    logger.info("Starting model evaluation...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Model Mean Squared Error (MSE): {mse}")
    logger.info(f"Model R2 Score: {r2}")

    return mse, r2

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from clean_data import clean_data
    from feature_engineering import feature_engineering
    from config import config

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
    X_test = test_df.drop(columns=[config['target_column']])
    y_test = test_df[config['target_column']]

    # Train a model
    model = Ridge()
    model.fit(X_train, y_train)

    # Evaluate the model
    mse, r2 = evaluate_model(model, X_test, y_test)
    print(f"Model Mean Squared Error (MSE): {mse}")
    print(f"Model R2 Score: {r2}")