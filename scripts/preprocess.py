import pandas as pd
import glob
import os

# Define the path to the raw data
data_path = "data/raw"

# Step 1: Load all CSV files with low_memory=False
all_files = glob.glob(os.path.join(data_path, "*.csv"))
df_list = [pd.read_csv(file, low_memory=False) for file in all_files]
df = pd.concat(df_list, ignore_index=True)

# Step 2: Drop irrelevant columns
df = df.drop(columns=['Unnamed: 0', 'rideable_type_duplicate_column_name_1'], errors='ignore')

# Step 3: Clean and preprocess
# Convert timestamps to datetime with mixed format support
df['started_at'] = pd.to_datetime(df['started_at'], format='mixed')
df['ended_at'] = pd.to_datetime(df['ended_at'], format='mixed')

# Drop rows with missing start_station_name
df = df.dropna(subset=['start_station_name'])

# Step 4: Aggregate by start station and hour
df['start_hour'] = df['started_at'].dt.floor('H')  # Round to nearest hour
trips_per_station_hour = df.groupby(['start_station_name', 'start_hour']).size().reset_index(name='trip_count')

# Step 5: Select top 3 stations by total trips
total_trips_per_station = df.groupby('start_station_name').size().reset_index(name='total_trips')
top_3_stations = total_trips_per_station.sort_values(by='total_trips', ascending=False).head(3)['start_station_name'].tolist()
print("Top 3 stations:", top_3_stations)

# Step 6: Filter data for top 3 stations and create a new DataFrame
trips_top_3 = trips_per_station_hour[trips_per_station_hour['start_station_name'].isin(top_3_stations)].copy()

# Step 7: Ensure data types are appropriate for Hopsworks
trips_top_3['start_station_name'] = trips_top_3['start_station_name'].astype(str)
trips_top_3['start_hour'] = trips_top_3['start_hour'].astype(str)  # Convert to string for Hopsworks compatibility
trips_top_3['trip_count'] = trips_top_3['trip_count'].astype(int)

# Step 8: Save the processed data locally
trips_top_3.to_csv('processed_trips_top_3.csv', index=False)

# Inspect the processed data
print("\nProcessed data (first 5 rows):")
print(trips_top_3.head())

import hopsworks
import pandas as pd
import time

# Step 1: Log in to Hopsworks and connect to the project
try:
    project = hopsworks.login(
        host="c.app.hopsworks.ai",
        project="CitiBikeTrip",
        api_key_value="DIrr083Keer9GlOI.3FUt2ZiErZwV9gDGP0i5fCINcaNopLE4YfPoswUh3HrGaepEZyMdO5VQkxFsl4d0"
    )
    print("Logged in to Hopsworks and connected to project 'CitiBikeTrip' successfully!")
except Exception as e:
    print(f"Failed to log in to Hopsworks or access project: {e}")
    raise

# Step 2: Connect to the Feature Store
fs = project.get_feature_store()

# Step 3: Delete the existing Feature Group (version 1) if it exists
try:
    fg = fs.get_feature_group(name="citi_bike_trips_fg", version=1)
    fg.delete()
    print("Existing Feature Group 'citi_bike_trips_fg' (version 1) deleted successfully!")
    # Wait a few seconds to allow backend synchronization
    time.sleep(10)  # Delay to ensure deletion propagates
except Exception as e:
    print(f"Failed to delete existing Feature Group or it doesn't exist: {e}")
    if "does not exist" in str(e):
        print("No existing Feature Group to delete, proceeding with creation.")

# Step 4: Load your processed data
df = pd.read_csv('processed_trips_top_3.csv')

# Step 5: Create a new Feature Group (version 1)
try:
    feature_group = fs.create_feature_group(
        name="citi_bike_trips_fg",
        version=1,
        description="Hourly trip counts for the top 3 Citi Bike stations",
        primary_key=["start_station_name", "start_hour"],
        online_enabled=True
    )
    print("Feature Group 'citi_bike_trips_fg' (version 1) created successfully!")
except Exception as e:
    print(f"Failed to create Feature Group: {e}")
    raise

# Step 6: Insert the data into the Feature Group
try:
    feature_group.insert(df)
    print("Data inserted into Feature Group 'citi_bike_trips_fg' (version 1) successfully!")
except Exception as e:
    print(f"Failed to insert data into Feature Group: {e}")
    raise

try:
    fg = fs.get_feature_group(name="citi_bike_trips_fg", version=1)
    df_retrieved = fg.read()
    print("\nFirst 5 rows of the uploaded data:")
    print(df_retrieved.head())
except Exception as e:
    print(f"Failed to read Feature Group data: {e}")
    raise
