import pandas as pd
import hopsworks
import os

# Step 1: Load raw data
print("Loading raw data...")
df = pd.read_csv('data/raw_trips.csv')
print("Raw data shape:", df.shape)

# Step 2: Preprocess data
df['starttime'] = pd.to_datetime(df['starttime'])
df['start_hour'] = df['starttime'].dt.floor('H')

# Select top 3 stations by trip count
top_stations = df['start_station_name'].value_counts().head(3).index
df = df[df['start_station_name'].isin(top_stations)].copy()
print("Data after filtering top 3 stations, shape:", df.shape)

# Aggregate trips by station and hour
df = df.groupby(['start_station_name', 'start_hour']).size().reset_index(name='trip_count')
print("Data after aggregation, shape:", df.shape)

# Step 3: Save processed data
df.to_csv('data/processed_trips_top_3.csv', index=False)
print("Processed data saved to 'data/processed_trips_top_3.csv'")

# Step 4: Log in to Hopsworks
print("Logging in to Hopsworks...")
project = hopsworks.login(
    host="c.app.hopsworks.ai",
    project="CitiBikeTrip",
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
print("Logged in successfully.")

# Step 5: Get Feature Store
fs = project.get_feature_store()
print("Feature Store retrieved:", fs)

# Step 6: Save to Feature Group
fg = fs.get_or_create_feature_group(
    name="citi_bike_trips_fg",
    version=1,
    description="Processed trip counts for top 3 stations",
    primary_key=['start_station_name', 'start_hour']
)
fg.insert(df, write_options={'wait': True})
print("Data inserted successfully into 'citi_bike_trips_fg'.")
