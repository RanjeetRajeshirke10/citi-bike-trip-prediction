import hopsworks
import mlflow
import os
import pandas as pd

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
lightgbm_full_run = runs[runs["tags.mlflow.runName"] == "LightGBM_Full"].iloc[-1]
mae = lightgbm_full_run["metrics.MAE"]

# Step 4: Hardcode the model path (update the run ID after running train.py)
model_path = r"C:\Users\ranje\Jupyter Notebooks\Sem 2\CDA ML\Citi_Bike_Trip\mlruns\473768752618069851\ed9b647c35784c1f96515b3d16c03b91\artifacts\model"
print(f"Hardcoded Model Path: {model_path}")
if os.path.exists(model_path):
    print("Model directory exists! Contents:", os.listdir(model_path))
else:
    print("Model directory does NOT exist!")
    raise FileNotFoundError(f"Model path {model_path} does not exist!")

# Step 5: Load the training data for input example
X_train = pd.read_csv('data/X_train.csv')

# Step 6: Register the model
model = mr.sklearn.create_model(
    name="citi_bike_trip_predictor",
    metrics={"mae": mae},
    description="Model to predict Citi Bike trip counts for top 3 stations using LightGBM Full",
    input_example=X_train.head(5),
    version=1
)

# Step 7: Save the model
model.save(model_path)

print(f"Model 'citi_bike_trip_predictor' (version 1) uploaded to Hopsworks with MAE: {mae}")
