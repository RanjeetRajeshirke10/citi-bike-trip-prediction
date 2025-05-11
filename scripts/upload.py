import hopsworks
import mlflow
import os
import pandas as pd
import urllib.parse

# Step 1: Log in to Hopsworks
project = hopsworks.login(
    host="c.app.hopsworks.ai",
    project="CitiBikeTrip",
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)

# Step 2: Get the Model Registry
mr = project.get_model_registry()

# Step 3: Get the latest run for LightGBM Full
runs = mlflow.search_runs(experiment_names=["CitiBikeModels"])
if runs.empty:
    raise ValueError("No runs found for experiment 'CitiBikeModels'. Ensure train.py ran successfully.")
lightgbm_full_run = runs[runs["tags.mlflow.runName"] == "LightGBM_Full"].iloc[-1]
mae = lightgbm_full_run["metrics.MAE"]

# Step 4: Fix the artifact path for the current environment
artifact_uri = lightgbm_full_run["artifact_uri"]
decoded_uri = urllib.parse.unquote(artifact_uri)
cleaned_uri = decoded_uri.replace("file://", "").lstrip(os.sep)
model_path = os.path.normpath(os.path.join(cleaned_uri, "model"))
print(f"Model Path: {model_path}")

# Verify the path exists
if os.path.exists(model_path):
    print("Model directory exists! Contents:", os.listdir(model_path))
else:
    print("Model directory does NOT exist! Check MLflow run artifacts.")
    raise FileNotFoundError(f"Model path {model_path} does not exist!")

# Step 5: Load the training data for input example
X_train = pd.read_csv('data/X_train.csv')

# Step 6: Register the model
model = mr.sklearn.create_model(
    name="citi_bike_trip_predictor",
    metrics={"mae": mae},
    description="Model to predict Citi Bike trip counts for top 3 stations using LightGBM Full",
    version=2  # Increment version since version 1 exists
)

# Step 7: Save the model
model.save(model_path)

print(f"Model 'citi_bike_trip_predictor' (version 2) uploaded to Hopsworks with MAE: {mae}")
