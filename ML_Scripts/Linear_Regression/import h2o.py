import h2o
from h2o.automl import H2OAutoML

# Initialize H2O
h2o.init()

# Load dataset
df = h2o.import_file("C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/data/BostonHousing.csv")

# Split data into train and test sets
train, test = df.split_frame(ratios=[0.8], seed=42)

# Define target and features
target = "medv"
features = [col for col in train.columns if col != target]

# Train AutoML model
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=features, y=target, training_frame=train)

# View leaderboard
lb = aml.leaderboard
print(lb)

# Predict on test set
predictions = aml.predict(test)
print(predictions)
