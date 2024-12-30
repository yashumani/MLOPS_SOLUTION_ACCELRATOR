# **Project: Comparative Analysis of Regression Models using PyCaret and FLAML**

---

## **Overview**

This project focuses on building an end-to-end pipeline for comparative regression analysis using two powerful frameworks: **PyCaret** and **FLAML**. The primary goal is to predict a target variable (e.g., graduation rate in the provided dataset) through robust data preprocessing, feature engineering, and hyperparameter optimization. By leveraging automation and machine learning, the pipeline identifies the best-performing model and provides actionable insights.

---

## **Purpose**

1. **Regression Model Comparison**:
   - Evaluate and compare the performance of PyCaret and FLAML in solving regression problems.
   - Identify the best model using metrics like R² and RMSE.

2. **End-to-End Machine Learning Pipeline**:
   - Automate processes including data cleaning, exploratory data analysis (EDA), feature engineering, and model evaluation.

3. **Hyperparameter Optimization**:
   - Use Optuna for efficient hyperparameter tuning and model selection.

4. **Visualization and Reporting**:
   - Generate insightful visualizations for model performance and a detailed EDA report using Sweetviz.

---

## **Features**

### 1. **Data Preprocessing**
   - Handles missing values using **Iterative Imputer (MICE)**.
   - Encodes categorical variables using one-hot encoding.
   - Identifies and removes outliers based on percentile thresholds.

### 2. **Exploratory Data Analysis (EDA)**
   - Generates a comprehensive Sweetviz report for understanding dataset characteristics.
   - Visualizes data distributions, correlations, and outliers.

### 3. **Regression Model Training**
   - **PyCaret**: Automates model selection, hyperparameter tuning, and evaluation.
   - **FLAML**: Lightweight and efficient AutoML framework for fast experimentation.

### 4. **Hyperparameter Tuning**
   - Uses **Optuna** for tuning hyperparameters dynamically across both PyCaret and FLAML.

### 5. **Model Evaluation**
   - Compares models using key metrics: R² (coefficient of determination) and RMSE (root mean squared error).
   - Provides detailed insights through visualizations such as scatter plots and residual distributions.

---

## **Setup and Requirements**

### **Environment**
Ensure you have a Python environment with the required dependencies installed. A recommended environment configuration is provided in the `environment.yml` file.

### **Dependencies**
The script requires the following libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `pycaret`
- `flaml`
- `optuna`
- `sweetviz`
- `scikit-learn`

### **Installation**
1. Clone this repository to your local machine.
2. Create a virtual environment:
   ```bash
   conda env create -f environment.yml
   conda activate env_MLOps
   ```
3. Install additional dependencies (if necessary) using `pip` or `conda`.

---

## **Pipeline Workflow**

1. **Data Collection**:
   - Loads the dataset from the specified path.
   - Logs information about the data.

2. **EDA**:
   - Generates a Sweetviz report for pre-cleaning and post-cleaning analysis.

3. **Data Cleaning**:
   - Removes duplicates and encodes categorical features.
   - Handles missing values using MICE.
   - Removes outliers using percentile-based thresholds.

4. **Feature Engineering**:
   - Prepares features for modeling (placeholder for custom feature engineering steps).

5. **Hyperparameter Tuning**:
   - Uses Optuna to tune parameters for PyCaret and FLAML.
   - Selects the best-performing model based on R².

6. **Model Evaluation**:
   - Builds and evaluates the best model using test data.
   - Visualizes performance through scatter plots and residual analysis.

7. **Reporting**:
   - Saves detailed logs and visualizations in the specified reports directory.

---

## **Execution**

Run the script using the following command:
```bash
python Linear_regression_1_5.py
```

---

## **Outputs**

1. **Logs**:
   - Detailed logs of every step stored in `logs.log`.

2. **Visualizations**:
   - Scatter plot of true vs predicted values.
   - Residual distribution plot.

3. **Sweetviz Report**:
   - Automatically generated HTML reports for EDA.

4. **Metrics**:
   - R² and RMSE scores for the best-performing model.

5. **Winner**:
   - Outputs the best-performing model name and its performance metrics.

---

## **Results**

### Example Output
| Metric        | PyCaret         | FLAML          |
|---------------|-----------------|----------------|
| R² (Accuracy) | **0.88**        | 0.85           |
| RMSE (Error)  | 2.34            | **2.30**       |

**Winner**: FLAML

---

## **Customizations**

1. **Feature Engineering**:
   - Add domain-specific feature transformations in the `feature_engineering` function.

2. **Hyperparameter Tuning**:
   - Modify the search space in the `objective` function to experiment with additional parameters.

3. **Visualization**:
   - Extend the visualization module for more detailed plots (e.g., feature importance, partial dependence).

