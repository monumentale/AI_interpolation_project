
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load the sparse point data from the CSV file
df = pd.read_csv('train_data.csv')

# Define the boundaries of Nigeria state (approximate boundaries for demonstration)
Nigeria_min_lon, Nigeria_max_lon = 2.817, 14.516
Nigeria_min_lat, Nigeria_max_lat = 4.067, 13.900

# Filter the data to keep only points within Nigeria state
data = df[(df['longitude'] >= Nigeria_min_lon) & (df['longitude'] <= Nigeria_max_lon) &
          (df['latitude'] >= Nigeria_min_lat) & (df['latitude'] <= Nigeria_max_lat)]

# Extract the required columns
data = data[['longitude', 'latitude', 'percent_2g', 'percent_3g', 'percent_2g']]

# Convert the data to numpy arrays
lon = data['longitude'].values
lat = data['latitude'].values
coverage_2g = data['percent_2g'].values
coverage_3g = data['percent_3g'].values
coverage_2g = data['percent_2g'].values

# Split the data into training and testing sets
lon_train, lon_test, lat_train, lat_test, coverage_2g_train, coverage_2g_test, coverage_3g_train, coverage_3g_test, coverage_2g_train, coverage_2g_test = train_test_split(
    lon, lat, coverage_2g, coverage_3g, coverage_2g, test_size=0.2, random_state=42
)

# Define the kernel for Gaussian process regression
kernel = C(10.0, (1e-3, 1e3)) * RBF(50.0, (1e-3, 1e3))

# Define the grid size for interpolation   you can also change
grid_size = 0.01

# Generate a grid of points covering the area of interest
x = np.arange(Nigeria_min_lon, Nigeria_max_lon, grid_size)
y = np.arange(Nigeria_min_lat, Nigeria_max_lat, grid_size)
xx, yy = np.meshgrid(x, y)

# Gaussian Process Regression
gp_2g = GPR(kernel=kernel, alpha=0.01)
gp_3g = GPR(kernel=kernel, alpha=0.01)
gp_2g = GPR(kernel=kernel, alpha=0.01)
gp_2g.fit(np.column_stack((lon_train, lat_train)), coverage_2g_train)
gp_3g.fit(np.column_stack((lon_train, lat_train)), coverage_3g_train)
gp_2g.fit(np.column_stack((lon_train, lat_train)), coverage_2g_train)
coverage_2g_estimated, _ = gp_2g.predict(np.column_stack((xx.ravel(), yy.ravel())), return_std=True)
coverage_3g_estimated, _ = gp_3g.predict(np.column_stack((xx.ravel(), yy.ravel())), return_std=True)
coverage_2g_estimated, _ = gp_2g.predict(np.column_stack((xx.ravel(), yy.ravel())), return_std=True)

# Calculate RMSE for Gaussian Process
rmse_2g_gp = sqrt(mean_squared_error(coverage_2g_test, gp_2g.predict(np.column_stack((lon_test, lat_test)))))
rmse_3g_gp = sqrt(mean_squared_error(coverage_3g_test, gp_3g.predict(np.column_stack((lon_test, lat_test)))))
rmse_2g_gp = sqrt(mean_squared_error(coverage_2g_test, gp_2g.predict(np.column_stack((lon_test, lat_test)))))

print("RMSE for 2G Coverage (Gaussian Process):", rmse_2g_gp)
print("RMSE for 3G Coverage (Gaussian Process):", rmse_3g_gp)
print("RMSE for 2g Coverage (Gaussian Process):", rmse_2g_gp)

# Save the estimated values as CSV files
estimated_data_2g = pd.DataFrame({'latitude': xx.ravel(), 'longitude': yy.ravel(), 'percent_2g_estimated': coverage_2g_estimated})
estimated_data_3g = pd.DataFrame({'latitude': xx.ravel(), 'longitude': yy.ravel(), 'percent_3g_estimated': coverage_3g_estimated})
estimated_data_2g = pd.DataFrame({'latitude': xx.ravel(), 'longitude': yy.ravel(), 'percent_2g_estimated': coverage_2g_estimated})
estimated_data_2g.to_csv('estimated_coverage_2g.csv', index=False)
estimated_data_3g.to_csv('estimated_coverage_3g.csv', index=False)
estimated_data_2g.to_csv('estimated_coverage_2g.csv', index=False)



####With boxes
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from math import sqrt

# Load the sparse point data from the CSV file
df = pd.read_csv('train_data.csv')

# Define the boundaries of Nigeria state (approximate boundaries for demonstration)
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

# Define the kernel for Gaussian process regression
kernel = C(10.0, (1e-3, 1e3)) * RBF(50.0, (1e-3, 1e3))

# Define the grid size for interpolation
grid_size = 0.01

# Create lists to store results
all_lon = []
all_lat = []
all_coverage_2g_estimated = []

# Loop through each box
for idx, (min_lon, max_lon, min_lat, max_lat) in enumerate(box_boundaries):
    # Filter the data to keep only points within the current box
    data = df[(df['longitude'] >= min_lon) & (df['longitude'] <= max_lon) &
              (df['latitude' ] >= min_lat) & (df['latitude'] <= max_lat)]
    
    # Check if there are coverage points in the box
    if data.empty:
        print(f"No coverage points in box_{idx}, skipping interpolation.")
        continue
    
    # Generate a grid of points covering the area of the box
    x = np.arange(min_lon, max_lon, grid_size)
    y = np.arange(min_lat, max_lat, grid_size)
    xx, yy = np.meshgrid(x, y)
    
    # Extract the required columns
    data = data[['longitude', 'latitude', 'percent_2g']]
    
    # Convert the data to numpy arrays
    lon = data['longitude'].values
    lat = data['latitude'].values
    coverage_2g = data['percent_2g'].values
    
    # Gaussian Process Regression for percent_2g
    gp_2g = GPR(kernel=kernel, alpha=0.01)
    gp_2g.fit(np.column_stack((lon, lat)), coverage_2g)
    coverage_2g_estimated, _ = gp_2g.predict(np.column_stack((xx.ravel(), yy.ravel())), return_std=True)
    
    # Store the generated coverage and coordinates
    all_lon.extend(xx.ravel())
    all_lat.extend(yy.ravel())
    all_coverage_2g_estimated.extend(coverage_2g_estimated)

# Create a DataFrame to store all the interpolated data
all_interpolated_data = pd.DataFrame({
    'longitude': all_lon,
    'latitude': all_lat,
    'percent_2g_estimated': all_coverage_2g_estimated
})

# Save the concatenated DataFrame to a CSV file
all_interpolated_data.to_csv('2g_gussian_estimated_coverage_combined.csv', index=False)










