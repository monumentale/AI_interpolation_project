import pandas as pd
import numpy as np
!pip install verde
import verde as vd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the data from the CSV file
df = pd.read_csv('train_data.csv')

# Define the boundaries of Nigeria state
Nigeria_min_lon, Nigeria_max_lon = 2.817, 14.516
Nigeria_min_lat, Nigeria_max_lat = 4.067, 13.900

# Define the grid of boxes
num_rows = 7
num_cols = 10
grid_resolution_x = (Nigeria_max_lon - Nigeria_min_lon) / num_cols
grid_resolution_y = (Nigeria_max_lat - Nigeria_min_lat) / num_rows

# Initialize an empty list to store box boundaries
box_boundaries = []

# Generate the boundaries for each box in the grid
for row in range(num_rows):
    for col in range(num_cols):
        box_min_lon = Nigeria_min_lon + col * grid_resolution_x
        box_max_lon = box_min_lon + grid_resolution_x
        box_min_lat = Nigeria_min_lat + row * grid_resolution_y
        box_max_lat = box_min_lat + grid_resolution_y
        box_boundaries.append((box_min_lon, box_max_lon, box_min_lat, box_max_lat))

# Define the coordinates, region, and spacing for interpolation
coordinates = (df['longitude'].values, df['latitude'].values)
spacing = 0.01

# Initialize an empty DataFrame to store the results
combined_results = pd.DataFrame()

# Loop through each box
for idx, (min_lon, max_lon, min_lat, max_lat) in enumerate(box_boundaries):
    # Filter the data to keep only points within the current box
    data = df[(df['longitude'] >= min_lon) & (df['longitude'] <= max_lon) &
              (df['latitude' ] >= min_lat) & (df['latitude' ] <= max_lat)]

    # Check if there is coverage data in the box
    if data.empty:
        print(f"No coverage points in box_{idx}, skipping interpolation.")
        continue

    # Extract longitude, latitude, and coverage values for the box
    lon_box = data['longitude'].values
    lat_box = data['latitude'].values
    coverage_2g_box = data['percent_2g'].values

    # Define the coordinates for the box
    coordinates_box = (lon_box, lat_box)

    # Create a chain for interpolation
    chain_box = vd.Chain(
        [
            ("mean", vd.BlockReduce(np.mean, spacing=spacing)),
            ("spline", vd.Spline()),
        ]
    )

    chain_box.fit(coordinates_box, coverage_2g_box)

    # Generate a grid using the fitted chain for the box
    grid_box = chain_box.grid(
        region=vd.get_region(coordinates_box),
        spacing=spacing,
        dims=["latitude", "longitude"],
        data_names="coverage_2g_estimated",
    )
    
    data_box = grid_box["coverage_2g_estimated"].values

    lats_box, lons_box = np.meshgrid(grid_box.latitude, grid_box.longitude)

    # Store the data in a DataFrame
    result_df = pd.DataFrame({
        'latitude': lats_box.ravel(),
        'longitude': lons_box.ravel(),
        'coverage_2g_estimated': data_box.ravel()
    })

    # Append the results to the combined DataFrame
    combined_results = combined_results.append(result_df, ignore_index=True)

# Save the combined DataFrame to a CSV file
combined_results.to_csv('2g_Spline_combined_coverage_estimates.csv', index=False)






