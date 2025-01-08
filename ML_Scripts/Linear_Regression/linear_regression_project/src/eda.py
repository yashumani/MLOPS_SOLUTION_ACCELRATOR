# FILE: /linear_regression_project/src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from generate_eda_report import generate_eda_report
from get_logger import get_logger
from config import config
import os

def perform_eda(df, title="EDA"):
    print("Executing eda.py")
    logger = get_logger('perform_eda')
    logger.info(f"--- {title} ---")

    eda_tool = config.get('eda_tool', {}).get('tool', 'sweetviz')
    open_browser = config.get('eda_tool', {}).get('open_browser', True)
    
    if eda_tool == 'dtale':
        import dtale
        logger.info("Generating EDA report using D-Tale...")
        d = dtale.show(df)
        if open_browser:
            d.open_browser()
        logger.info("D-Tale EDA report generated successfully.")
    elif eda_tool == 'ydata-profiling':
        from ydata_profiling import ProfileReport
        logger.info("Generating EDA report using ydata-profiling...")
        profile = ProfileReport(df, title=title)
        profile_path = os.path.join(config['reports_path'], f"{title}_ydata_profiling_report.html")
        profile.to_file(profile_path)
        logger.info(f"ydata-profiling EDA report saved to {profile_path}")
    else:
        generate_eda_report(df, title)
    
    logger.info("Generating box plot for numeric columns...")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df[numeric_columns])
        plt.title('Box Plot of Numeric Columns')
        plt.xlabel('Columns')
        plt.ylabel('Values')
        plt.savefig(os.path.join(config['reports_path'], f"{title}_box_plot_numeric_columns.png"))
        plt.close()
        logger.info("Box plot generated successfully.")
    return df