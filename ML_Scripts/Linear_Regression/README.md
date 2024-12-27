# Linear Regression Workflow with Data Exploration and Model Evaluation

## Overview
This project provides a comprehensive automated pipeline for performing linear regression analysis. The workflow includes data loading, cleaning, exploratory data analysis (EDA), feature engineering, model training, evaluation, and visualization. It supports multiple regression models including Ridge, Lasso, ElasticNet, and standard Linear Regression.

---

## Features
1. **Data Handling**:
   - Load data from a CSV file.
   - Handle missing values and outliers using predefined methods.

2. **Exploratory Data Analysis (EDA)**:
   - Generate summary statistics.
   - Create visualizations such as correlation heatmaps and target variable distributions.
   - Save EDA results as images and CSV files.

3. **Preprocessing**:
   - Apply polynomial feature expansion.
   - Perform Variance Inflation Factor (VIF) analysis to identify multicollinearity.
   - Standardize features for regression models.

4. **Model Training and Evaluation**:
   - Train and evaluate multiple regression models: Linear Regression, Ridge, Lasso, and ElasticNet.
   - Compute metrics such as Mean Squared Error (MSE) and R-squared (R²).
   - Perform cross-validation to ensure model robustness.

5. **Visualization**:
   - Plot actual vs. predicted values with residuals for better interpretability.
   - Visualize training vs. validation errors to detect overfitting or underfitting.

6. **Narrative Generation**:
   - Generate recommendations for model improvement based on performance metrics.

---

## Installation
1. Clone this repository or copy the script.
2. Install required Python packages:

   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn statsmodels
   ```

---

## Usage

### 1. Configure Constants
Modify the following constants in the script to match your setup:
- **`DATA_PATH`**: Path to the CSV file containing the dataset.
- **`TARGET_COLUMN`**: The name of the target variable column.
- **`REPORTS_PATH`**: Path to save the generated reports and visualizations.
- **`DEGREE`**: Polynomial degree for feature expansion.
- **`ALPHA`**: Regularization strength for Ridge, Lasso, and ElasticNet models.

### 2. Run the Script
Execute the script in your terminal or IDE:

```bash
python <script_name>.py
```

### 3. Select the Best Model
After the model comparison, you will be prompted to select the best-performing model. Enter one of the following options:
- `linear`
- `ridge`
- `lasso`
- `elasticnet`

The script will provide narratives and save predictions for the selected model.

---

## Workflow Details

### Data Cleaning
- Handles missing values by removing rows with null values.
- Removes outliers based on the 1st and 99th percentiles of numeric features.

### Exploratory Data Analysis (EDA)
- Saves:
  - Correlation heatmap
  - Target variable distribution
  - Summary statistics as CSV files

### Feature Engineering
- Generates polynomial features for non-linear relationships.
- Evaluates multicollinearity using VIF analysis.
- Scales features for effective model training.

### Model Training and Evaluation
- Trains four models: Linear Regression, Ridge, Lasso, and ElasticNet.
- Reports:
  - Mean Squared Error (MSE)
  - R-squared (R²)
  - Cross-validation R-squared scores
- Visualizes:
  - Actual vs. Predicted values
  - Training vs. Validation error

### Narrative Generation
Provides:
- Key observations of model performance
- Recommendations for improvement

---

## File Outputs
- **`Dataset_Summary.csv`**: Descriptive statistics of the dataset.
- **`Correlation_Heatmap.png`**: Heatmap of feature correlations.
- **`Target_Variable_Distribution.png`**: Histogram of the target variable.
- **`VIF_Analysis.csv`**: Variance Inflation Factor results.
- **`<model_type>_Actual_vs_Predicted_Improved.png`**: Visualization of actual vs. predicted values.
- **`<model_type>_Training_vs_Validation_Error.png`**: Training and validation error comparison.
- **`<model_type>_Predictions.csv`**: Predicted values for the selected model.

---

## Example Output
### Model Comparison:
```
--- Model Comparison ---
Model: Linear
Mean Squared Error: 24.5321
R-squared: 0.7524
------------------------------
Model: Ridge
Mean Squared Error: 25.1238
R-squared: 0.7489
------------------------------
Model: Lasso
Mean Squared Error: 27.9823
R-squared: 0.7322
------------------------------
Model: ElasticNet
Mean Squared Error: 26.5123
R-squared: 0.7408
------------------------------
```

### Recommendations for Improvement:
```
--- Key Observations ---
Model Performance:
- The R-squared value on the test set is 0.7524, indicating the model explains a high proportion of the variance.
- The Mean Squared Error (MSE) is 24.5321, indicating the average squared difference between actual and predicted values.
- Cross-validation R-squared scores: [0.752, 0.746, 0.750, 0.755, 0.760]

Recommendations for Improvement:
- Use Ridge, Lasso, or ElasticNet regression to handle multicollinearity.
- Check for potential feature interactions and add non-linear terms if needed.
- Regularly validate the model on unseen data to ensure robustness.
```

---

## License
This project is open-source and can be used, modified, and distributed freely.

---

## Author
**Yashu Sharma**  
For inquiries or collaborations, feel free to contact.
