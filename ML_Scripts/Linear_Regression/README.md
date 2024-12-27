### **README.md**

---

# **Linear Regression with PyCaret and FLAML**

## **Overview**
This project demonstrates the use of **PyCaret** and **FLAML** for linear regression to predict housing prices using the Boston Housing dataset. The script implements a complete workflow that includes:

1. **Data Cleaning**: Handles missing values, removes duplicates, and identifies outliers using percentile-based thresholds.
2. **Model Training**:
   - **PyCaret** for automated regression model comparison and hyperparameter tuning.
   - **FLAML** for efficient and lightweight AutoML with hyperparameter tuning using **Optuna**.
3. **Model Evaluation**: Compares the performance of the two frameworks using metrics like R² (coefficient of determination) and RMSE (root mean squared error).
4. **Model Comparison**: Identifies the best-performing framework.

---

## **Prerequisites**

1. **Python**: Version 3.10 or higher.
2. **Required Libraries**:
   - pandas
   - matplotlib
   - seaborn
   - pycaret
   - flaml
   - optuna
   - scikit-learn
   - numpy

---

## **Setup**

### **Conda Environment**

```bash
# Create a new conda environment
conda create -n env_linear_regression python=3.10 -y

# Activate the environment
conda activate env_linear_regression

# Install dependencies
pip install pandas matplotlib seaborn pycaret flaml optuna scikit-learn numpy
```

---

## **How to Run the Script**

1. **Download the Dataset**:
   - Place the `BostonHousing.csv` file in the specified path in the script (`C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/data/BostonHousing.csv`).
   - Ensure the target column (`medv`) matches the dataset.

2. **Execute the Script**:
   ```bash
   python Linear_regression_1_3.py
   ```

3. **Output**:
   - The script will clean the dataset, train models using PyCaret and FLAML, and display the R² and RMSE for both frameworks.
   - A summary of the winning model based on R² will be displayed.

---

## **Theoretical Summary**

### **Linear Regression**
Linear regression is a supervised learning technique used for predicting continuous outcomes. It fits a linear equation to the observed data. The goal is to minimize the sum of squared residuals between the predicted and actual values.

---

### **Steps in the Script**

1. **Data Cleaning**:
   - Handles missing values by replacing them with the median.
   - Removes duplicates.
   - Identifies and removes outliers based on the 1st and 99th percentiles of feature distributions.

2. **PyCaret Workflow**:
   - **Setup**: Configures the dataset for model comparison.
   - **Model Comparison**: Selects the best regression model.
   - **Hyperparameter Tuning**: Optimizes the selected model's performance.
   - **Prediction and Evaluation**: Evaluates the tuned model on the test set using R² and RMSE.

3. **FLAML Workflow**:
   - **Model Training**: Automatically selects and trains the best-performing regression model within a specified time budget.
   - **Prediction and Evaluation**: Evaluates the selected model on the test set using R² and RMSE.
   - **Hyperparameter Optimization**: Uses Optuna to further optimize hyperparameters.

4. **Comparison**:
   - Compares R² and RMSE for models trained using PyCaret and FLAML.
   - Declares the winner based on the highest R².

---

### **Metrics**
1. **R² (Coefficient of Determination)**:
   Measures how well the model explains the variability of the target variable.
   - **Range**: 0 to 1 (higher is better).

2. **RMSE (Root Mean Squared Error)**:
   Measures the average prediction error.
   - **Lower RMSE indicates better model performance.**

---

## **Example Output**

```plaintext
--- Data Cleaning ---
Dataset shape after cleaning: (490, 14)

--- PyCaret Linear Regression ---
Best Model from PyCaret (Pre-Tuning):
<LinearRegression object>
Tuned Model from PyCaret:
<LinearRegression object>
PyCaret R²: 0.82
PyCaret RMSE: 3.45

--- FLAML Linear Regression ---
Best Model from FLAML (Pre-Tuning):
lgbm
Best Config: {'n_estimators': 100, ...}
Best Loss: 0.32
FLAML R²: 0.81
FLAML RMSE: 3.48

--- Model Comparison ---
PyCaret R²: 0.82, RMSE: 3.45
FLAML R²: 0.81, RMSE: 3.48

Winner: PyCaret
```

---

## **Conclusion**

This script showcases the power of automated machine learning frameworks like **PyCaret** and **FLAML** for solving regression problems. Both frameworks offer efficient model selection and tuning capabilities. The side-by-side comparison provides insights into their strengths and weaknesses.
