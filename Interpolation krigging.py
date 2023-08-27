import pandas as pd
import geopandas as gpd
from pykrige.ok import OrdinaryKriging
import numpy as np
import matplotlib.pyplot as plt
from pyproj import CRS

# Import CSV file of point data
df_points = pd.read_csv('train_data.csv')

# Convert point data to GeoDataFrame and set CRS to WGS84
gdf_points = gpd.GeoDataFrame(
    df_points, geometry=gpd.points_from_xy(df_points.longitude, df_points.latitude))
gdf_points.crs = CRS.from_epsg(4326)

# Import shapefile and set CRS to WGS84
shapefile_path = 'nga_admbnda_adm0_osgof_20190417.shp'
gdf_shapefile = gpd.read_file(shapefile_path)
gdf_shapefile = gdf_shapefile.to_crs(epsg=4326)

# Set up Kriging model
OK = OrdinaryKriging(
    gdf_points['longitude'].values,
    gdf_points['latitude'].values,
    gdf_points['percent_coverage'].values,
    variogram_model='linear',
    verbose=False,
    enable_plotting=False)

# Create grid for interpolation
x_min, y_min, x_max, y_max = gdf_shapefile.total_bounds
x_range = np.linspace(x_min, x_max, 100)
y_range = np.linspace(y_min, y_max, 100)
grid_x, grid_y = np.meshgrid(x_range, y_range)
grid_z, grid_var = OK.execute('grid', grid_x.flatten(), grid_y.flatten())

# Convert interpolated data to GeoDataFrame and set CRS to WGS84
gdf_interp = gpd.GeoDataFrame(
    {'latitude': grid_y.flatten(), 'longitude': grid_x.flatten(), 'percent_coverage': grid_z},
    geometry=gpd.points_from_xy(grid_x.flatten(), grid_y.flatten()))
gdf_interp.crs = CRS.from_epsg(4326)

# Overlay interpolated data on shapefile
fig, ax = plt.subplots(figsize=(10, 10))
gdf_shapefile.plot(ax=ax, color='white', edgecolor='black')
gdf_interp.plot(ax=ax, column='percent_coverage', cmap='viridis', alpha=0.5)

# Add title and legend
ax.set_title('National Mobile Network Coverage')
ax.legend(['Shapefile', 'Interpolated Coverage'])

# Show plot
plt.show()
