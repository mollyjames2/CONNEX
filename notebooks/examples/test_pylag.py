import geopandas as gpd
from connex.analysis.analysis import open_trajectory_data,summarize_connectivity_start_end, summarize_connectivity_by_path
from connex.plot.plot import plot_trajectories, plot_kde_snapshot, plot_kde_snapshot_with_nodes, start_node_assignment
from connex.plot.plot import  plot_kde, plot_kde_with_nodes, plot_connectivity_graph_map
from connex.analysis.graph_network import build_connectivity_matrix_start_end,build_connectivity_matrix_by_path, connectivity_graph, summarize_connectivity_graph_metrics
import xarray as xr
from datetime import timedelta

# --- User Configuration ---
data_path = "data/pylag_1.nc"  # or .nc
outputdt = timedelta(seconds=300) # the output timestep of the simulation data
settlement_hours = 48  # Minimum age to settle
extent = [-5.1, -4.9, 50, 50.2]#[min_lon, max_lon, min_lat, max_lat]

time_var = "time"
time_dim = "time"
particle_dim = "particles"
lon_var = "longitude"
lat_var = "latitude"
pld = None  #pld in days. setting to None will use the last timestep in the
         #simulation data, so will setting to a larger number than the days in the simulation


# --- Choose what to show ---
trajectories = True
snapshot = True # will show connectivity at a set end point (set by pld) - if pld is None this will be the last timestep in the datafile
comp_window = True  # will show connectivity based on all particle locations during their competency winoe
dispersal_cloud = True #kde plots of the dispersal clouds

# --- Load trajectory data into an xarray Dataset ---
ds = open_trajectory_data(data_path)





if trajectories:
    # --- Plot particle trajectories ---
    # This visualizes individual particle paths, optionally colored by release node
    print("\n Plotting trajectories...")
    plot_trajectories(
        data_path=data_path,
        pld=pld,  
        outputdt=outputdt,  
        extent=None,
        show_nodes=False,
        time_dim=time_dim, 
        particle_dim=particle_dim,
        lon_var=lon_var,
        lat_var=lat_var,
        save_path=None
    )

if snapshot:
    if dispersal_cloud:

       # --- Plot dispersal cloud at given PLD for all particles ---
       # Shows dispersal density for all particles at a selected PLD (in days)
       plot_kde_snapshot(
           data_path=data_path,
           outputdt=outputdt,
           pld=pld,
           extent=extent,
           time_var=time_var,
           time_dim=time_dim,
           particle_dim=particle_dim,
           lon_var=lon_var,
           lat_var=lat_var,
           show_nodes=False,
           show_particles=False
       )
               
if comp_window:
    if dispersal_cloud:
        # --- Plot overall KDE for all particles during their competency window ---
        # This generates a single density cloud using all particle positions
        # from settlement time until their PLD (if provided), or until the last timestep.
        plot_kde(
            data_path=data_path,
            outputdt=outputdt,
            settlement_hours=settlement_hours,
            extent=extent,
            pld_days=pld,
            time_var=time_var,
            time_dim=time_dim,
            particle_dim=particle_dim,
            lon_var=lon_var,
            lat_var=lat_var,
            show_nodes=False,
            show_particles=False
        )
       
