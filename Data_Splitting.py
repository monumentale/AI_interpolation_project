import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the data from the CSV file
df = pd.read_csv('NG_mobile coverage (1).csv')

lon = df['longitude'].values
lat = df['latitude'].values
coverage_2g = df['percent_2g'].values
coverage_3g = df['percent_3g'].values
coverage_4g = df['percent_4g'].values

# Split the data into training and test sets
lon_train, lon_test, lat_train, lat_test, coverage_2g_train, coverage_2g_test, coverage_3g_train, coverage_3g_test, coverage_4g_train, coverage_4g_test = train_test_split(
    lon, lat, coverage_2g, coverage_3g, coverage_4g, test_size=0.2, random_state=42
)

# Create DataFrames for training and test data
train_data = pd.DataFrame({
    'longitude': lon_train,
    'latitude': lat_train,
    'percent_2g': coverage_2g_train,
    'percent_3g': coverage_3g_train,
    'percent_4g': coverage_4g_train
})

test_data = pd.DataFrame({
    'longitude': lon_test,
    'latitude': lat_test,
    'percent_2g': coverage_2g_test,
    'percent_3g': coverage_3g_test,
    'percent_4g': coverage_4g_test
})

# Save the training and test data as separate CSV files
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)