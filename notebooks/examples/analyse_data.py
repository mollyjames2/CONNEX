import geopandas as gpd
from connex.analysis.analysis import open_trajectory_data,summarize_connectivity_start_end, summarize_connectivity_by_path
from connex.plot.plot import plot_trajectories, plot_kde_snapshot, plot_kde_snapshot_with_nodes
from connex.plot.plot import  plot_kde, plot_kde_with_nodes
from connex.analysis.graph_builder import build_connectivity_matrix_start_end, build_connectivity_matrix_by_path
import xarray as xr
from datetime import timedelta

# --- User Configuration ---
data_path = "data/example_trajectories.zarr"  # or .nc
shapefile_path = "data/node_shp.shp"
start_time = None  # Or "2025-01-01T00:00:00"
end_time = None    # Or "2025-01-05T00:00:00"
outputdt = timedelta(hours=6) # the output timestep of the simulation data
settlement_hours = 0  # Minimum age to settle
time_var = "time"
time_dim = "obs"
particle_dim = "trajectory"
lon_var = "lon"
lat_var = "lat"
ppn = 100 # particles per node
pld = None  #pld in days. setting to None will use the last timestep in the
         #simulation data, so will setting to a larger number than the days in the simulation
elapsed_time= None #timedelta(days=2) # choose the timestep for kde_plots. None will
                               #plot the last timestep in the simulation 


# Load the polygons from shapefile
# These polygons define spatial nodes (e.g., settlement or habitat regions)
polygon_gdf = gpd.read_file(shapefile_path)
node_polys = polygon_gdf.geometry.tolist()
node_ids = polygon_gdf["node_id"].tolist()

# --- Load trajectory data into an xarray Dataset ---
ds = open_trajectory_data(data_path)

# --- Plot particle trajectories ---
# This visualizes individual particle paths, optionally colored by release node
print("\n🔹 Plotting trajectories...")
plot_trajectories(
    data_path=data_path,
    extent=None,  # or [min_lon, max_lon, min_lat, max_lat]
    show_nodes=True,
    node_polys=node_polys,
    start_time=start_time,
    end_time=end_time,
    time_var=time_var,
    time_dim=time_dim,
    particle_dim=particle_dim,
    color_by_node=True,
    particles_per_node=ppn,
    save_path=None  # or ".png" to save
)


# --- Run start-end connectivity summary ---
print("\n🔹 Calculating start-end connectivity summary...")
summary = summarize_connectivity_start_end(
    ds=ds,
    particles_per_node=ppn,
    node_ids=node_ids,
    node_polys=node_polys,
    lon_var=lon_var,
    lat_var=lat_var,
    time_var=time_var,
    time_dim=time_dim,
    particle_dim=particle_dim,
    pld_days=pld
)

# --- Run path-based connectivity summary ----
# this allows conectivty to be considered if a particle passes through a node
# in its competency window
print("\n🔹 Calculating path based connectivity summary...")
summary = summarize_connectivity_by_path(
    ds=ds,
    particles_per_node=ppn,
    node_ids=node_ids,
    node_polys=node_polys,
    settlement_hours=settlement_hours,
    pld_days=pld,
    outputdt=outputdt,
    lon_var=lon_var,
    lat_var=lat_var,
    time_var=time_var,
    time_dim=time_dim,
    particle_dim=particle_dim
)


# --- Start-End Matrix ---
# Returns a node-by-node matrix counting final destinations from each source
print("\n🔷 Start-End Connectivity Matrix:")
start_end_matrix = build_connectivity_matrix_start_end(
    data_path=data_path,
    shapefile_path=shapefile_path,
    pld_days=pld,
    time_var=time_var,
    time_dim=time_dim,
    particle_dim=particle_dim
)
print(start_end_matrix)

# --- Path-Based Matrix ---
# Captures node-to-node transitions during competency window (not just start/end)
print("\n🔶 Path-Based Connectivity Matrix:")
path_matrix = build_connectivity_matrix_by_path(
    data_path=data_path,
    shapefile_path=shapefile_path,
    settlement_hours=settlement_hours,
    pld_days=pld,
    outputdt=outputdt,
    time_var=time_var,
    time_dim=time_dim,
    particle_dim=particle_dim,
    lon_var=lon_var,
    lat_var=lat_var
)
print(path_matrix)


# --- Plot dispersal cloud at given timepoint for all particles ---
# Shows dispersal density for all particles at a selected elapsed time
plot_kde_snapshot(
    data_path=data_path,
    outputdt=outputdt,
    elapsed_time=elapsed_time,
    time_var=time_var,
    time_dim=time_dim,
    particle_dim=particle_dim,
    lon_var=lon_var,
    lat_var=lat_var,
    show_nodes=True,
    node_polys=node_polys
)

# --- Plot dispersal cloud at given timepoint by release node ---
#  Shows separate dispersal density clouds for particles grouped by release node
plot_kde_snapshot_with_nodes(
    data_path=data_path,
    outputdt=outputdt,
    elapsed_time=elapsed_time,
    particles_per_node=ppn,
    time_var=time_var,
    time_dim=time_dim,
    particle_dim=particle_dim,
    lon_var=lon_var,
    lat_var=lat_var,
    show_nodes=True,
    node_polys=node_polys
)

# --- Plot overall KDE for all particles during their competency window ---
# This generates a single density cloud using all particle positions
# from settlement time until their PLD (if provided), or until the last timestep.
plot_kde(
    data_path=data_path,
    outputdt=outputdt,
    settlement_hours=settlement_hours,
    pld_days=pld,
    time_var=time_var,
    time_dim=time_dim,
    particle_dim=particle_dim,
    lon_var=lon_var,
    lat_var=lat_var,
    show_nodes=True,
    node_polys=node_polys
)

# --- Plot node-wise KDEs during the competency window ---
# This creates separate KDE clouds for each release node using particle positions 
# within their competency window (from settlement time to PLD or end of sim).
# Useful for visualizing spatial spread by origin.
plot_kde_with_nodes(
    data_path=data_path,
    outputdt=outputdt,
    settlement_hours=settlement_hours,
    pld_days=pld,
    particles_per_node=ppn,
    time_var=time_var,
    time_dim=time_dim,
    particle_dim=particle_dim,
    lon_var=lon_var,
    lat_var=lat_var,
    show_nodes=True,
    node_polys=node_polys
)

