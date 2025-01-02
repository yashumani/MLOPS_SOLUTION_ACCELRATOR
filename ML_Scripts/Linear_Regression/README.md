# **End-to-End Machine Learning Workflow with Enhanced Feature Engineering and Hyperparameter Tuning**

---

## **Overview**

This project implements an end-to-end machine learning workflow for regression tasks, including data loading, exploratory data analysis (EDA), data cleaning, feature engineering, hyperparameter tuning, and model evaluation. The script leverages advanced techniques such as outlier detection, enhanced feature engineering with GridSearchCV, and hyperparameter optimization using Optuna.

---

## **Features**

1. **Outlier Detection**:
   - Univariate and multivariate outlier detection using z-score, IQR, and machine learning models like Isolation Forest and One-Class SVM.

2. **Enhanced Feature Engineering**:
   - Normalization using StandardScaler.
   - Advanced feature engineering with GridSearchCV for optimal transformations and model selection.

3. **Performance Visualization**:
   - Generates visualizations for EDA, model performance, and residual analysis.

---

## **Requirements**

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `pycaret`
- `flaml`
- `optuna`
- `sweetviz`
- `scikit-learn`
- `fuzzywuzzy`
- `shutil`

---

## **Setup Instructions**

1. Clone the repository:
   ```bash
   git clone https://github.com/SAVYMINDS/YS_MVP.git
   cd YS_MVP
   ```

2. Create and activate a virtual environment:
   ```bash
   conda env create -f environment.yml
   conda activate env_MLOps
   ```

3. Ensure all dependencies are installed:
   ```bash
   conda install --file requirements.txt
   ```

---

## **Usage**

1. **Specify the dataset path and target column**:
   - Update `DATA_PATH` and `TARGET_COLUMN` in the script with the appropriate values.

2. **Run the script**:
   ```bash
   python Linear_regression_1_5.py
   ```

3. **Generated Reports and Visualizations**:
   - Sweetviz reports and visualizations are saved in the `Reports` directory.

---

## **Process Workflow**

1. **Data Loading**:
   - Loads the dataset and checks for compatibility with regression tasks.

2. **Exploratory Data Analysis (EDA)**:
   - Generates Sweetviz reports and visualizations for initial data insights.

3. **Data Cleaning**:
   - Handles missing values, removes duplicates, and detects anomalies.

4. **Feature Engineering**:
   - Applies normalization and advanced feature engineering with GridSearchCV.

5. **Hyperparameter Tuning**:
   - Optimizes model hyperparameters using Optuna.

6. **Model Evaluation**:
   - Evaluates models using multiple metrics and visualizes performance.

---

## **Performance Metrics**

- **R² (Coefficient of Determination)**
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **MAPE (Mean Absolute Percentage Error)**
- **Explained Variance**

---

## **Reports and Outputs**

1. **Logs**:
   - Execution logs stored in `logs.log`.

2. **Visualizations**:
   - Scatter plot of true vs predicted values.
   - Residual distribution plot.

3. **EDA Reports**:
   - Pre-cleaning and post-cleaning Sweetviz reports.

4. **Metrics**:
   - R², RMSE, MAE, MAPE, and Explained Variance scores for model evaluation.

5. **Best Model**:
   - Prints the best-performing model's name and metrics.

---

## **Example Usage**

```bash
python Linear_regression_1_5.py
```

**Interpreting Results**:
- The script will output the best model based on R² and RMSE scores.
- Visualizations and reports will be saved in the `Reports` directory.

---

## **Limitations and Future Improvements**

- **Limitations**:
  - The current implementation may not handle all types of datasets.
  - Further tuning and validation are recommended for robustness.

- **Future Improvements**:
  - Incorporate additional feature engineering techniques.
  - Extend support for more machine learning models.

---

## **Contact Information**

For queries, please contact: [Your Email]

---

