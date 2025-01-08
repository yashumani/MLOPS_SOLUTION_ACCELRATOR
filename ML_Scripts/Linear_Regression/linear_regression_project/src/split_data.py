# FILE: /linear_regression_project/src/split_data.py

from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
from get_logger import get_logger
from config import config

def split_data(df, config):
    print("Executing split_data.py")
    logger = get_logger('split_data')
    logger.info("--- Data Splitting ---")
    split_type = config.get('split_type', 'train_test')
    if split_type == 'train_test':
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
        logger.info("Performed train/test split.")
    elif split_type == 'k_fold':
        kf = KFold(n_splits=config.get('n_splits', 5), shuffle=True, random_state=123)
        for train_index, test_index in kf.split(df):
            train_df, test_df = df.iloc[train_index], df.iloc[test_index]
        logger.info("Performed K-Fold split.")
    elif split_type == 'time_series':
        tscv = TimeSeriesSplit(n_splits=config.get('n_splits', 5))
        for train_index, test_index in tscv.split(df):
            train_df, test_df = df.iloc[train_index], df.iloc[test_index]
        logger.info("Performed time-series split.")
    else:
        raise ValueError("Invalid split type specified in configuration.")
    return train_df, test_df