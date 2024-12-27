Below is the **`README.md`** file for the provided Python script:

---

# **Linear Regression Comparison: PyCaret vs. FLAML**

This project aims to demonstrate an end-to-end comparison between **PyCaret** and **FLAML** for linear regression tasks. The dataset used is the **Boston Housing dataset**, which contains information about house prices and associated features.

---

## **Table of Contents**
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Features](#features)
- [Code Workflow](#code-workflow)
- [Execution](#execution)
- [Results](#results)
- [Theoretical Summary](#theoretical-summary)

---

## **Prerequisites**
Ensure you have Python 3.10+ installed. You will also need the following libraries:
- **PyCaret**: For automated regression modeling and hyperparameter tuning.
- **FLAML**: For lightweight and fast automated machine learning.
- **Scikit-learn**: For metric evaluation and data splitting.
- **Matplotlib** and **Seaborn**: For visualization.

---

## **Installation**
### **Using Conda**
```bash
conda create --name env_linear_regression python=3.10
conda activate env_linear_regression
conda install --file requirements.txt
```

### **Using Pip**
```bash
pip install -r requirements.txt
```

---

## **Features**
1. **Data Cleaning**:
   - Duplicate removal.
   - Filling missing values using the median.
   - Outlier removal using the 1st and 99th percentiles.

2. **Modeling and Tuning**:
   - **PyCaret**: Automated model selection, hyperparameter tuning, and predictions.
   - **FLAML**: Lightweight automated machine learning with fast execution and custom hyperparameter tuning.

3. **Evaluation Metrics**:
   - **R²** (Coefficient of Determination): Measures model accuracy.
   - **RMSE** (Root Mean Squared Error): Measures error magnitude.

4. **Comparison**:
   - Both models are compared based on R² and RMSE to identify the better-performing approach.

---

## **Code Workflow**
### **1. Data Loading**
The dataset is loaded from the specified CSV path:
```python
DATA_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/data/BostonHousing.csv"
```

### **2. Data Cleaning**
- Remove duplicates.
- Handle missing values by replacing them with the median.
- Remove outliers using percentile thresholds.

### **3. PyCaret Linear Regression**
- Setup the PyCaret environment using the training data.
- Identify the best model using `compare_models()`.
- Tune the best model using `tune_model()`.
- Evaluate the model on the test data.

### **4. FLAML Linear Regression**
- Use the `AutoML` class from FLAML to train a regression model.
- Evaluate the best model on the test data.
- Perform hyperparameter optimization using custom configurations.

### **5. Model Comparison**
- Compare the results of PyCaret and FLAML based on R² and RMSE.
- Determine the better-performing model.

---

## **Execution**
Run the Python script:
```bash
python Linear_regression_1_3.py
```

### **Expected Output**
1. Data Cleaning Summary:
   - Initial and post-cleaning dataset shapes.
2. PyCaret Results:
   - Best model (pre-tuning and post-tuning).
   - Evaluation metrics (R² and RMSE).
3. FLAML Results:
   - Best model details.
   - Evaluation metrics (R² and RMSE).
4. Model Comparison:
   - Final comparison of PyCaret and FLAML with winner identification.

---

## **Results**
- **PyCaret**:
  - Tuned model performance on the test dataset.
  - Provides insights into model comparison and evaluation.

- **FLAML**:
  - Offers fast and efficient model training with optimal configurations.
  - Competitive performance with lightweight execution.

---

## **Theoretical Summary**
1. **PyCaret**:
   - A low-code machine learning library designed to simplify regression workflows.
   - Includes robust built-in hyperparameter tuning and model comparison features.
   - Ideal for users seeking quick and accurate model training with minimal coding.

2. **FLAML**:
   - Lightweight AutoML library focused on fast and efficient model training.
   - Optimized for scenarios with limited computational resources.
   - Provides flexibility with customizable hyperparameter tuning through integration with **Optuna**.

3. **Comparison**:
   - **PyCaret** is more comprehensive with rich diagnostic outputs, while **FLAML** focuses on efficiency.
   - PyCaret’s built-in tuning can yield better models for complex datasets, whereas FLAML excels in speed and simplicity.

---

## **Conclusion**
This script provides a streamlined approach to comparing **PyCaret** and **FLAML** for linear regression tasks. By integrating automated workflows with detailed metrics, users can efficiently determine the best-performing model for their dataset.

---
