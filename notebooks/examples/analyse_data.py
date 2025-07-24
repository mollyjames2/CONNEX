import geopandas as gpd
from connex.analysis.analysis import open_trajectory_data,summarize_connectivity_start_end, summarize_connectivity_by_path
from connex.plot.plot import plot_trajectories
from connex.analysis.graph_builder import build_connectivity_matrix_start_end
import xarray as xr

filepath = 'data/example_trajectories.zarr'

# Load the polygons from shapefile
polygon_gdf = gpd.read_file("data/node_shp.shp")
node_polys = polygon_gdf.geometry.tolist()
node_ids = polygon_gdf["node_id"].tolist()

ds = open_trajectory_data(filepath)

# Plot particle trajectories
plot_trajectories(
    data_path=filepath,
    show_nodes=True,
    node_polys=node_polys,
#    start_time="2002-05-07",
#    end_time="2002-05-08",
    color_by_node=True,
    particles_per_node=100,
    time_var="obs",
    particle_var="trajectory"
)

# Print a node connectivity summary based on the start and end locations of particles
summary = summarize_connectivity_start_end(
    ds,
    particles_per_node=100,
    node_ids=node_ids ,
    node_polys=node_polys,
    time_dim = 'obs',
    particle_dim = 'trajectory'#,
#    start_time="2002-05-07",
#    end_time="2002-05-08",
)

# Print a node connectivity cummary based on the paths of the particles
summary = summarize_connectivity_by_path(
    ds,
    particles_per_node=100,
    node_ids=node_ids,
    node_polys=node_polys,
    time_dim="obs",
    particle_dim="trajectory",
    settlement_hours=1  # Only consider connectivity within first 48 hours
)


matrix = build_connectivity_matrix_start_end(
    data_path=filepath,
    shapefile_path="data/node_shp.shp",
#    start_time="2002-05-07",
#    end_time="2002-05-08",
    time_var="obs",
    particle_var="trajectory"
)

print(matrix)

