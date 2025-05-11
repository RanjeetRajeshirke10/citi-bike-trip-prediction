import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import mlflow
import mlflow.sklearn
import os
import urllib.parse

# Step 1: Load the processed data
df = pd.read_csv('processed_trips_top_3.csv')
df['start_hour'] = pd.to_datetime(df['start_hour'])

# Step 2: Create lag features efficiently
all_station_data = []
for station in df['start_station_name'].unique():
    station_data = df[df['start_station_name'] == station].sort_values('start_hour').copy()
    lag_columns = [station_data['trip_count'].shift(lag) for lag in range(1, 673)]
    lag_df = pd.concat([station_data] + [pd.Series(col, name=f'lag_{lag}') for lag, col in enumerate(lag_columns, 1)], axis=1)
    all_station_data.append(lag_df)

df = pd.concat(all_station_data, ignore_index=True)
df = df.dropna()

# Step 3: Prepare features and target
X = df[[f'lag_{i}' for i in range(1, 673)]]
y = df['trip_count']

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Step 5: Set MLflow experiment
mlflow.set_experiment("CitiBikeModels")

# Model 1: Baseline (Mean)
with mlflow.start_run(run_name="Baseline"):
    mean_prediction = np.mean(y_train)
    baseline_predictions = np.full_like(y_test, mean_prediction)
    mae = mean_absolute_error(y_test, baseline_predictions)
    mlflow.log_metric("MAE", mae)
    print(f"Baseline MAE: {mae}")

# Model 2: LightGBM with 28-day lags
with mlflow.start_run(run_name="LightGBM_Full") as run:
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mlflow.log_metric("MAE", mae)
    mlflow.sklearn.log_model(model, "model", input_example=X_train.head(5))
    print(f"LightGBM Full MAE: {mae}")
    # Debug: Fix path and verify
    artifact_uri = run.info.artifact_uri
    decoded_uri = urllib.parse.unquote(artifact_uri)  # Decode %20 to spaces
    cleaned_uri = decoded_uri.replace("file://", "").lstrip(os.sep)  # Remove leading separator
    model_path = os.path.normpath(os.path.join(cleaned_uri, "model"))  # Normalize path
    print(f"Artifact URI for LightGBM_Full: {artifact_uri}")
    print(f"Model Path for LightGBM_Full: {model_path}")
    if os.path.exists(model_path):
        print("Model directory exists! Contents:", os.listdir(model_path))
    else:
        print("Model directory does NOT exist!")

# Model 3: LightGBM with feature reduction (top 10 features)
selector = SelectKBest(score_func=f_regression, k=10)
X_train_reduced = selector.fit_transform(X_train, y_train)
X_test_reduced = selector.transform(X_test)

with mlflow.start_run(run_name="LightGBM_Reduced"):
    model = GradientBoostingRegressor()
    model.fit(X_train_reduced, y_train)
    predictions = model.predict(X_test_reduced)
    mae = mean_absolute_error(y_test, predictions)
    mlflow.log_metric("MAE", mae)
    mlflow.sklearn.log_model(model, "model", input_example=X_train_reduced[:5])
    print(f"LightGBM Reduced MAE: {mae}")
