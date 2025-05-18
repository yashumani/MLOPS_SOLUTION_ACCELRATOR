# FILE: /ML_Scripts/Linear_Regression/linear_regression_project/src/model_selection.py

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, r2_score
from config import config
from get_logger import get_logger

def select_model(train_df, config):
    logger = get_logger('model_selection')
    logger.info("Starting model selection...")

    # Split the training data into features and target
    X_train = train_df.drop(columns=[config['target_column']])
    y_train = train_df[config['target_column']]

    # Define the model
    model = Ridge()

    # Define the parameter grid
    param_grid = config['grid_search']['param_grid']

    # Define the scoring metric
    scoring = make_scorer(r2_score)

    # Perform grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_
    logger.info(f"Best model parameters: {grid_search.best_params_}")
    logger.info(f"Best model R2 score: {grid_search.best_score_}")

    return best_model

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from clean_data import clean_data
    from feature_engineering import feature_engineering

    # Load dataset
    df = pd.read_csv(config['data_path'])

    # Clean the dataset
    df = clean_data(df)

    # Perform feature engineering
    df = feature_engineering(df, config)

    # Split dataset into training and testing subsets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Select the best model
    best_model = select_model(train_df, config)
    print(f"Best model: {best_model}")