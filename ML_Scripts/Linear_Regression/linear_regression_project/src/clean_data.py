# FILE: /linear_regression_project/src/clean_data.py

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from Levenshtein import distance as levenshtein_distance
from get_logger import get_logger

def clean_data(df):
    print("Executing clean_data.py")
    logger = get_logger('clean_data')
    logger.info("--- Data Cleaning ---")
    initial_shape = df.shape

    # Remove duplicates
    df = df.drop_duplicates()
    logger.info(f"Shape after removing duplicates: {df.shape}")

    print("Column types:\n", df.dtypes)
    print("Top 5 rows of the dataset:\n", df.head())

    # Data Deduplication with Levenshtein Distance
    logger.info("Performing data deduplication with Levenshtein Distance...")
    text_cols = df.select_dtypes(include=['object']).columns
    threshold = 3  # Set an appropriate threshold for your use case
    for col in text_cols:
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x))
        df[col] = df[col].astype(str)

        unique_values = df[col].unique()
        for i, value1 in enumerate(unique_values):
            for value2 in unique_values[i+1:]:
                if levenshtein_distance(value1, value2) < threshold:
                    df[col] = df[col].replace(value2, value1)

    logger.info("Shape after deduplication: %s", df.shape)

    # Impute missing values
    logger.info("Performing advanced imputation using KNN...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    imputer = KNNImputer()
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    logger.info("Shape after imputation: %s", df.shape)

    return df