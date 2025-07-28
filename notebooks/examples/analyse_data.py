import geopandas as gpd
from connex.analysis.analysis import open_trajectory_data,summarize_connectivity_start_end, summarize_connectivity_by_path
from connex.plot.plot import plot_trajectories, plot_kde_snapshot, plot_kde_snapshot_with_nodes, start_node_assignment
from connex.plot.plot import  plot_kde, plot_kde_with_nodes, plot_connectivity_graph_map
from connex.analysis.graph_network import build_connectivity_matrix_start_end,build_connectivity_matrix_by_path, connectivity_graph, summarize_connectivity_graph_metrics
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
pld = None  #pld in days. setting to None will use the last timestep in the
         #simulation data, so will setting to a larger number than the days in the simulation


# --- Choose what to show ---
trajectories = True
snapshot = True # will show connectivity at a set end point (set by pld) - if pld is None this will be the last timestep in the datafile
comp_window = True # will show connectivity based on all particle locations during their competency winoe
connectivity_summary = True # summarises connectivity between nodes (number of particles retained, lost outside the node networl, and reaching other nodes)
dispersal_cloud = True #kde plots of the dispersal clouds
network_analysis = True #graph network analysis
plot_connectivity = True # plot of graph network analysis


# Load the polygons from shapefile
# These polygons define spatial nodes (e.g., settlement or habitat regions)
polygon_gdf = gpd.read_file(shapefile_path)
node_polys = polygon_gdf.geometry.tolist()
node_ids = polygon_gdf["node_id"].tolist()

# --- Load trajectory data into an xarray Dataset ---
ds = open_trajectory_data(data_path)

# --- Calculate the node that each particle starts in --- 
#(required for plotting trajectories and dispersal clouds by nodes, and calculating connectivity summaries)
start_nodes = start_node_assignment(
    data_path=data_path,
    node_polys=node_polys,
    node_ids=node_ids,
    time_var="time",
    time_dim="obs",
    particle_dim="trajectory",
    lon_var="lon",
    lat_var="lat"
)



if trajectories:
    # --- Plot particle trajectories ---
    # This visualizes individual particle paths, optionally colored by release node
    print("\n Plotting trajectories...")
    plot_trajectories(
        data_path=data_path,
        pld=pld,  
        outputdt=outputdt,  
        extent=None,
        show_nodes=True,
        node_polys=node_polys,
        time_var=time_var,
        time_dim=time_dim,
        particle_dim=particle_dim,
        lon_var=lon_var,
        lat_var=lat_var,
        color_by_node=True,
        start_nodes=start_nodes,  
        save_path=None
    )


if snapshot:
    # --- Run start-end connectivity summary ---
    print("\n Calculating start-end connectivity summary...")
    summary = summarize_connectivity_start_end(
        ds=ds,
        start_nodes=start_nodes,
        node_ids=node_ids,
        node_polys=node_polys,
        lon_var=lon_var,
        lat_var=lat_var,
        time_var=time_var,
        time_dim=time_dim,
        particle_dim=particle_dim,
        pld_days=pld
    )

    if dispersal_cloud:
  
       # --- Plot dispersal cloud at given PLD for all particles ---
       # Shows dispersal density for all particles at a selected PLD (in days)
       plot_kde_snapshot(
           data_path=data_path,
           outputdt=outputdt,
           pld=pld,  
           time_var=time_var,
           time_dim=time_dim,
           particle_dim=particle_dim,
           lon_var=lon_var,
           lat_var=lat_var,
           show_nodes=True,
           node_polys=node_polys
       )
  
       # --- Plot dispersal cloud at given PLD by release node ---
       # Shows separate dispersal density clouds for particles grouped by release node
       plot_kde_snapshot_with_nodes(
           data_path=data_path,
           outputdt=outputdt,
           pld=pld,
           time_var=time_var,
           time_dim=time_dim,
           particle_dim=particle_dim,
           lon_var=lon_var,
           lat_var=lat_var,
           show_nodes=True,
           start_nodes=start_nodes,
           node_ids=node_ids,
           node_polys=node_polys
       )


    if network_analysis:
        # --- Start-End Matrix ---
        # Returns a node-by-node matrix counting final destinations from each source
        print("\n🔷 Start-End Connectivity Matrix:")
        matrix = build_connectivity_matrix_start_end(
            data_path=data_path,
            shapefile_path=shapefile_path,
            pld_days=pld,
            time_var=time_var,
            time_dim=time_dim,
            particle_dim=particle_dim
        )
        print(matrix)
        
        # --- Graph Network Analysis of the start end matrix ---
        #graph analysis using FastGreedy (undirected, no self-recruitment)
        metrics_df, graph = connectivity_graph(
            matrix,
            community_algorithm="fastgreedy",
            directed=False,
            remove_self_loops=True
        )
        
        print(metrics_df)
        
        if connectivity_summary:
            # --- Summarising connectivity based on the start and end particle locations ---
            # Provide output_file = 'file.txt' to save results as a text file
            summarize_connectivity_graph_metrics(metrics_df)
            
        if plot_connectivity:
            # --- Plotting the connectivity graph
            plot_connectivity_graph_map(
                graph=graph,
                shapefile_path=shapefile_path,
                title="Connectivity Graph (FastGreedy, Directed, No Self-Loops)",
                node_size=10,
                edge_color="black",
                edge_alpha=0.5,
                edge_width_scale=0.25,    # make edge weights more visible
                show_labels=True,
                show_arrows=True,
                save_path=None
            )

            
            
if comp_window:
    # --- Run path-based connectivity summary ----
    # this allows conectivty to be considered if a particle passes through a node
    # in its competency window
    print("\n Calculating path based connectivity summary...")
    summary = summarize_connectivity_by_path(
        ds=ds,
        start_nodes=start_nodes,
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

    if dispersal_cloud:
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
            start_nodes=start_nodes,
            time_var=time_var,
            time_dim=time_dim,
            particle_dim=particle_dim,
            lon_var=lon_var,
            lat_var=lat_var,
            show_nodes=True,
            node_polys=node_polys
        )

    if network_analysis:
        # --- Path-Based Matrix ---
        # Captures node-to-node transitions during competency window (not just start/end)
        print("\n🔶 Path-Based Connectivity Matrix:")
        matrix = build_connectivity_matrix_by_path(
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
        print(matrix)

        # --- Graph Network Analysis of the competency window  matrix ---
        #graph analysis using FastGreedy (undirected, no self-recruitment)
        metrics_df, graph = connectivity_graph(
            matrix,
            community_algorithm="fastgreedy",
            directed=False,
            remove_self_loops=True
        )
        
        print(metrics_df)
        
        if connectivity_summary:
            # --- Summarising connectivity based on the start and particle positions during
            # their competency window ---
            # Provide output_file = 'file.txt' to save results as a text file
            summarize_connectivity_graph_metrics(metrics_df)
            
        if plot_connectivity:
            
            # --- Plotting the connectivity graph
            # Plot the network on a map
            plot_connectivity_graph_map(
                graph=graph,
                shapefile_path=shapefile_path,
                title="Connectivity Graph (FastGreedy, Directed, No Self-Loops)",
                node_size=10,
                edge_color="black",
                edge_alpha=0.5,
                edge_width_scale=0.25,    # make edge weights more visible
                show_labels=True,
                show_arrows=True,
                save_path=None
            )



