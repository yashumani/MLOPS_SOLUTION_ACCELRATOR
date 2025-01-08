from get_logger import get_logger
from config import config

def domain_based_imputation(df):
    logger = get_logger('domain_based_imputation')
    logger.info("Performing domain-based imputation...")
    if config.get('domain') == 'healthcare' and 'age' in df.columns:
        median_age = df['age'].median()
        df['age'].fillna(median_age, inplace=True)
    logger.info("Domain-based imputation completed.")
    return df