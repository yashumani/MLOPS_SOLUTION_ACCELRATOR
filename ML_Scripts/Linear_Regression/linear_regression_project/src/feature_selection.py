# FILE: /linear_regression_project/src/feature_selection.py

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from get_logger import get_logger

def feature_selection(df, target_column, method='variance_threshold', threshold=0.1):
    print("Executing feature_selection.py")
    logger = get_logger('feature_selection')
    logger.info("Performing feature selection using method: %s", method)

    if method == 'variance_threshold':
        selector = VarianceThreshold(threshold)
        features = df.drop(columns=[target_column])
        selected_features = features.loc[:, selector.fit(features).get_support()]
        df = pd.concat([selected_features, df[target_column]], axis=1)
    elif method == 'correlation_threshold':
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(pd.np.triu(pd.np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        to_drop = [col for col in to_drop if col != target_column]  # Ensure target column is not dropped
        df = df.drop(columns=to_drop)
    else:
        logger.error("Invalid feature selection method specified: %s", method)
        raise ValueError("Invalid feature selection method specified.")

    logger.info("Feature selection completed. Shape after feature selection: %s", df.shape)
    print("Columns after feature selection:", df.columns)
    return df