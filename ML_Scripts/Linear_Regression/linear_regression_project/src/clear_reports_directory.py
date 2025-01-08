# FILE: /linear_regression_project/src/clear_reports_directory.py

import os
import shutil
from get_logger import get_logger

def clear_reports_directory(path):
    print("Executing clear_reports_directory.py")
    logger = get_logger('clear_reports_directory')
    logger.info("Clearing the Reports directory...")
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.error(f"Failed to clear file: {file_path}. Reason: {e}")