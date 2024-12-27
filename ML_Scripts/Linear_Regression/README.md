### **Updated `README.md` for `Linear_regression_1_3.py`**

---

# **Linear Regression Comparison: PyCaret vs. FLAML**

This project implements an end-to-end solution to compare the performance of **PyCaret** and **FLAML** in building a linear regression model for the **Boston Housing Dataset**. The script includes data cleaning, exploratory data analysis, model training, hyperparameter tuning, and performance evaluation.

---

## **Table of Contents**
- [Overview](#overview)
- [File Information](#file-information)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Workflow Summary](#workflow-summary)
- [Execution](#execution)
- [Logging](#logging)
- [Visualization](#visualization)
- [Results](#results)
- [Theoretical Insights](#theoretical-insights)

---

## **Overview**
The script demonstrates:
1. Data Cleaning: Duplicate removal, missing value imputation, and outlier handling.
2. Exploratory Data Analysis (EDA): Visualizations and summary statistics to understand data patterns.
3. Model Training and Hyperparameter Tuning:
   - **PyCaret**: Automated model selection and tuning.
   - **FLAML**: Lightweight and efficient AutoML solution with additional hyperparameter tuning using **Optuna**.
4. Model Evaluation: Comparison of R² and RMSE scores to identify the better-performing model.
5. Visualization: Insights into model performance through scatter plots and residual analysis.

---

## **File Information**
**Filename:** `Linear_regression_1_3.py`  
**Purpose:** Compare the performance of PyCaret and FLAML for linear regression tasks on the Boston Housing dataset.

---

## **Prerequisites**
Ensure the following libraries are installed in your environment:
- `pandas`
- `matplotlib`
- `seaborn`
- `pycaret`
- `flaml`
- `optuna`
- `scikit-learn`

---

## **Installation**

### **Using Conda**
```bash
conda create --name env_linear_regression python=3.10
conda activate env_linear_regression
conda install -c conda-forge pandas matplotlib seaborn pycaret flaml optuna scikit-learn
```

### **Using Pip**
```bash
pip install pandas matplotlib seaborn pycaret flaml optuna scikit-learn
```

---

## **Workflow Summary**

### **1. Exploratory Data Analysis (EDA)**
- Summary statistics (`info()` and `describe()`).
- Visualizations:
  - Histograms for feature distributions.
  - Correlation heatmaps.

### **2. Data Cleaning**
- **Duplicates**: Removed to ensure dataset integrity.
- **Missing Values**: Imputed with column medians.
- **Outliers**: Filtered using 1st and 99th percentiles for all numerical columns.

### **3. PyCaret Linear Regression**
- **Setup**: Initializes the PyCaret environment.
- **Model Comparison**: Identifies the best-performing model.
- **Hyperparameter Tuning**: Optimizes the selected model using `R²` as the performance metric.
- **Evaluation**: Computes R² and RMSE on the test dataset.

### **4. FLAML Linear Regression**
- **AutoML Training**: Automatically identifies the best-performing regression model.
- **Evaluation**: Predicts outcomes on the test dataset and calculates R² and RMSE.

### **5. Model Comparison**
- Both models are compared based on R² and RMSE.
- The better-performing model is visualized.

---

## **Execution**
Run the script as follows:
```bash
python Linear_regression_1_3.py
```

### **Expected Outputs**
1. Logs (`logs.log`) with details about data cleaning, EDA, model performance, and comparison.
2. Visualizations for EDA and model performance.
3. Printed R² and RMSE scores for both models.

---

## **Logging**
- All operations, including data cleaning, model training, and evaluation, are logged into a file named `logs.log`.
- Example log entries:
  ```
  2024-12-27 11:30:00 - INFO - Dataset loaded successfully.
  2024-12-27 11:35:10 - INFO - Initial dataset shape: (506, 14)
  2024-12-27 11:40:20 - INFO - PyCaret R²: 0.82, RMSE: 4.53
  ```

---

## **Visualization**
The script generates the following visualizations:
1. **EDA**:
   - Histograms for feature distributions.
   - Correlation heatmap.
2. **Model Performance**:
   - Scatter plot of true vs predicted values.
   - Residuals distribution.

---

## **Results**
### Example Model Comparison
| Metric        | PyCaret         | FLAML          |
|---------------|-----------------|----------------|
| R² (Accuracy) | **0.82**        | 0.80           |
| RMSE (Error)  | 4.53            | **4.40**       |

Winner: **FLAML**

---

## **Theoretical Insights**
### **PyCaret**
- Comprehensive, low-code library with built-in diagnostics.
- Allows quick prototyping of regression models.
- Includes hyperparameter tuning and detailed model comparisons.

### **FLAML**
- Lightweight AutoML library optimized for speed and efficiency.
- Automatically selects the best model configuration with minimal computational overhead.
- Ideal for scenarios requiring fast experimentation.

---

## **Conclusion**
The script demonstrates a streamlined workflow for linear regression tasks using automated tools. By comparing **PyCaret** and **FLAML**, users can identify the best-fit approach based on accuracy, error metrics, and computational efficiency.