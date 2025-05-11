import hopsworks
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import mlflow.pyfunc
import os

# Step 1: Log in to Hopsworks
project = hopsworks.login(
    host="c.app.hopsworks.ai",
    project="CitiBikeTrip",
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)

# Step 2: Get the Feature Store
fs = project.get_feature_store()

# Step 3: Load the latest data from the Feature Group
fg = fs.get_feature_group(name="citi_bike_trips_fg", version=1)
df = fg.read()
df['start_hour'] = pd.to_datetime(df['start_hour'])

# Step 4: Create lag features (same as training)
all_station_data = []
for station in df['start_station_name'].unique():
    station_data = df[df['start_station_name'] == station].sort_values('start_hour').copy()
    lag_columns = [station_data['trip_count'].shift(lag) for lag in range(1, 673)]
    lag_df = pd.concat([station_data] + [pd.Series(col, name=f'lag_{lag}') for lag, col in enumerate(lag_columns, 1)], axis=1)
    all_station_data.append(lag_df)

df = pd.concat(all_station_data, ignore_index=True)
df = df.dropna()
X = df[[f'lag_{i}' for i in range(1, 673)]]

# Step 5: Load the deployed model from Hopsworks
mr = project.get_model_registry()
model = mr.get_model("citi_bike_trip_predictor", version=2)
model_dir = model.download()
loaded_model = mlflow.pyfunc.load_model(model_dir)

# Step 6: Generate predictions
predictions = loaded_model.predict(X)

# Step 7: Add predictions to the DataFrame
df['predicted_trip_count'] = predictions

# Step 8: Save predictions to a new Feature Group or dataset
prediction_fg = fs.get_or_create_feature_group(
    name="citi_bike_predictions_fg",
    version=1,
    description="Predicted trip counts for top 3 stations",
    primary_key=['start_station_name', 'start_hour', 'predicted_trip_count']
)
prediction_fg.insert(df[['start_station_name', 'start_hour', 'predicted_trip_count']], write_options={'wait': True})

print("Predictions generated and saved to Hopsworks Feature Group.")
