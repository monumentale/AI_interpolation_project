import numpy as np
from scipy.spatial import cKDTree
import pandas as pd

# Read CSV containing longitudes, latitudes, and network coverage info
df = pd.read_csv('train_data.csv')

Nigeria_min_lon, Nigeria_max_lon = 2.817, 14.516
Nigeria_min_lat, Nigeria_max_lat = 4.067, 13.900

# Filter the data to keep only points within Nigeria state
data = df[(df['longitude'] >= Nigeria_min_lon) & (df['longitude'] <= Nigeria_max_lon) &
          (df['latitude'] >= Nigeria_min_lat) & (df['latitude'] <= Nigeria_max_lat)]

# Define the boundaries of your study area
min_longitude = data['longitude'].min()
max_longitude = data['longitude'].max()
min_latitude = data['latitude'].min()
max_latitude = data['latitude'].max()

# Define the grid resolution (spacing between points)
grid_resolution = 0.01  # Choose an appropriate value

# Generate a grid of new longitudes and latitudes
new_longitudes = np.arange(min_longitude, max_longitude + grid_resolution, grid_resolution)
new_latitudes = np.arange(min_latitude, max_latitude + grid_resolution, grid_resolution)

# Create a list of all combinations of longitudes and latitudes
new_points = [(longitude, latitude) for latitude in new_latitudes for longitude in new_longitudes]

# Create a KDTree for efficient nearest neighbor search
tree = cKDTree(data[['longitude', 'latitude']].values)

# Parameters for IDW
power = 2
num_neighbors = 3

# Perform interpolation for each new point
interpolated_values = []
for new_longitude, new_latitude in new_points:
    distances, indices = tree.query([new_longitude, new_latitude], k=num_neighbors)
    weights = 1.0 / (distances ** power)
    interpolated_value = np.sum(weights * data['percent_2g'].values[indices]) / np.sum(weights)
    interpolated_values.append(interpolated_value)

# Create a DataFrame with the interpolated values and coordinates
interpolated_data = pd.DataFrame({
    'longitude': [point[0] for point in new_points],
    'latitude': [point[1] for point in new_points],
    'interpolated_value': interpolated_values
})

# Write the interpolated data to a new CSV file
interpolated_data.to_csv('2g_Idw_interpolated_data.csv', index=False)
