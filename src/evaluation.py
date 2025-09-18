
"""
Evaluation module.

Compares models and identifies the best recipe using the specified primary metric.
"""

import pandas as pd
from typing import Dict, Any, Tuple


def rank_models(models: Dict[str, Any], primary_metric: str, task: str) -> Tuple[str, Any]:
    """Rank models based on primary metric and return the best model key and object.

    Args:
        models (Dict[str, Any]): Model results.
        primary_metric (str): Metric used for ranking.
        task (str): Task type.

    Returns:
        Tuple[str, Any]: Best model key and model object.
    """
    best_score = None
    best_key = None

    for key, result in models.items():
        if 'error' in result:
            continue
        if key == 'pycaret':
            res_df = result.get('results')
            if res_df is None:
                continue
            if primary_metric in res_df.columns:
                score = res_df.loc[0, primary_metric]
            else:
                continue
        else:
            continue  # Skip ranking FLAML due to missing direct metric
        if best_score is None or score > best_score:
            best_score = score
            best_key = key
    best_model = models.get(best_key, {}).get('best_model') if best_key else None
    return best_key, best_model
