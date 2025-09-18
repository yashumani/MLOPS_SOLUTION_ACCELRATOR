
"""
Data ingestion module.

Responsible for loading datasets from local files or remote locations (e.g., Azure ML). This version focuses on local CSV ingestion.
"""

import pandas as pd
from typing import Tuple


def load_data(path: str) -> Tuple[pd.DataFrame, str]:
    """Load dataset from a given path.

    Args:
        path (str): Path to the CSV file.

    Returns:
        Tuple[pd.DataFrame, str]: Loaded DataFrame and dataset name.
    """
    try:
        df = pd.read_csv(path)
        dataset_name = path.split('/')[-1]
        return df, dataset_name
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")
