import os
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.cm as cm
from datetime import timedelta
from scipy.stats import gaussian_kde
import geopandas as gpd   
from matplotlib.patches import FancyArrowPatch
import matplotlib.transforms as mtransforms

def plot_trajectories(
    data_path,
    extent=None,
    show_nodes=False,
    node_polys=None,
    save_path=None,
    start_time=None,
    end_time=None,
    time_var="time",
    time_dim="obs",
    particle_dim="trajectory",
    color_by_node=False,
    particles_per_node=None
):
    """
    Plot particle trajectories over time from trajectory datasets.

    Supports NetCDF, Zarr, and CSV formats. Each particle's path is plotted from start
    to end with optional markers at initial and final locations.

    Parameters:
    - data_path: Path to the dataset (.zarr, .nc, or .csv)
    - extent: Optional [min_lon, max_lon, min_lat, max_lat] for map view
    - show_nodes: Whether to draw node polygons
    - node_polys: List of shapely Polygon geometries for node outlines
    - save_path: If set, plot is saved to this file instead of shown
    - start_time, end_time: Optional datetime strings for filtering trajectories
    - color_by_node: If True, colors particles by source node
    - particles_per_node: Number of particles released per node (for color logic)

    Notes:
    - Particles are grouped by index to assign node-based coloring
    - Adds start (o) and end (x) markers for each trajectory
    """
    
    print(f" Reading data from: {data_path}")
    ext = os.path.splitext(data_path)[-1].lower()
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # --- Color setup ---
    colors = None
    if color_by_node and particles_per_node:
        if ext in [".zarr", ".nc"]:
            ds_temp = xr.open_zarr(data_path) if ext == ".zarr" else xr.open_dataset(data_path)
            num_particles = ds_temp.sizes[particle_dim]
        elif ext == ".csv":
            df_temp = pd.read_csv(data_path)
            num_particles = df_temp["particle_id"].nunique()
        else:
            raise ValueError("Unsupported file format for coloring.")

        num_nodes = num_particles // particles_per_node
        cmap = cm.get_cmap("tab10", num_nodes)
        colors = [cmap(i) for i in range(num_nodes)]

    # --- Load and filter data ---
    if ext in [".zarr", ".nc"]:
        ds = xr.open_zarr(data_path) if ext == ".zarr" else xr.open_dataset(data_path)

        # Time filtering using the variable, not the dimension
        times = pd.to_datetime(ds[time_var].values)
        time_mask = np.full(times.shape, True)
        if start_time:
            time_mask &= times >= pd.to_datetime(start_time)
        if end_time:
            time_mask &= times <= pd.to_datetime(end_time)
        time_idxs = np.where(time_mask)[0]

        if len(time_idxs) == 0:
            raise ValueError("No data in specified time range.")

        # Determine correct time axis
        time_axis = ds[time_var].dims[0]

        lons = ds["lon"].isel({time_axis: time_idxs}).values
        lats = ds["lat"].isel({time_axis: time_idxs}).values

        # Transpose if needed to [particles, time]
        if ds["lon"].dims[0] == time_dim:
            lons = lons.transpose()
            lats = lats.transpose()

        for i in range(lons.shape[0]):
            color = colors[i // particles_per_node] if color_by_node and colors else "blue"
            ax.plot(lons[i], lats[i], linewidth=0.8, alpha=0.6, transform=ccrs.PlateCarree(), color=color)
            ax.plot(lons[i, 0], lats[i, 0], marker='o', color=color, markersize=3, transform=ccrs.PlateCarree())
            ax.plot(lons[i, -1], lats[i, -1], marker='x', color=color, markersize=3, transform=ccrs.PlateCarree())

    elif ext == ".csv":
        df = pd.read_csv(data_path)
        df["time"] = pd.to_datetime(df["time"])
        if start_time:
            df = df[df["time"] >= pd.to_datetime(start_time)]
        if end_time:
            df = df[df["time"] <= pd.to_datetime(end_time)]

        for pid, group in df.groupby("particle_id"):
            color = colors[int(pid) // particles_per_node] if color_by_node and colors else "blue"
            ax.plot(group["lon"], group["lat"], linewidth=0.8, alpha=0.6, transform=ccrs.PlateCarree(), color=color)
            ax.plot(group["lon"].iloc[0], group["lat"].iloc[0], marker='o', color=color, markersize=3, transform=ccrs.PlateCarree())
            ax.plot(group["lon"].iloc[-1], group["lat"].iloc[-1], marker='x', color=color, markersize=3, transform=ccrs.PlateCarree())

    else:
        raise ValueError("Unsupported file format. Use .zarr, .nc, or .csv")

    # --- Optional node polygons ---
    if show_nodes and node_polys:
        for poly in node_polys:
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, color='red', linewidth=1.2, transform=ccrs.PlateCarree())

    ax.coastlines()
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    # Manually add axis labels
    ax.text(0.5, -0.1, 'Longitude', transform=ax.transAxes,ha='center', va='top', fontsize=12)
    ax.text(-0.08, 0.5, 'Latitude', transform=ax.transAxes,ha='center', va='bottom', rotation='vertical', fontsize=12)

    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    plt.title("Particle Trajectories (Colored by Source Node)" if color_by_node else "Particle Trajectories")
    ax.plot([], [], marker='o', color='black', linestyle='None', label='Start')
    ax.plot([], [], marker='x', color='black', linestyle='None', label='End')
    ax.legend(loc='lower left')

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Plot saved to {save_path}")
    else:
        plt.show()



def plot_kde_snapshot(
    data_path,
    outputdt,
    elapsed_time=None,
    time_var="time",
    time_dim="obs",
    particle_dim="trajectory",
    lon_var="lon",
    lat_var="lat",
    extent=None,
    save_path=None,
    show_nodes=False,
    node_polys=None
):
    """
    Plot a KDE (kernel density estimate) cloud for all particles at a specific timestep.

    This function gives a spatial density view of particle positions at a selected
    elapsed time since release. Useful for visualizing general dispersal envelopes.

    Parameters:
    - data_path: Path to dataset (.zarr, .nc, or .csv)
    - outputdt: Timedelta of model output intervals (e.g., timedelta(hours=6))
    - elapsed_time: Time since release to extract position snapshot (or None for last timestep)
    - show_nodes: Whether to overlay node polygons
    - node_polys: List of shapely Polygon geometries
    - extent: Optional [min_lon, max_lon, min_lat, max_lat] view
    - save_path: Path to save the plot instead of displaying
    - time_var, time_dim, lon_var, lat_var: Variable and dimension names to support flexible datasets

    Notes:
    - Adds KDE cloud and raw particle points
    - Node outlines are optionally plotted with one legend label
    """
    
    ext = os.path.splitext(data_path)[-1].lower()

    if ext in [".zarr", ".nc"]:
        ds = xr.open_zarr(data_path) if ext == ".zarr" else xr.open_dataset(data_path)
        num_steps = ds.sizes[time_dim]
        target_index = num_steps - 1 if elapsed_time is None else int(elapsed_time / outputdt)

        lons = ds[lon_var].isel({time_dim: target_index}).values
        lats = ds[lat_var].isel({time_dim: target_index}).values

        if ds[lon_var].dims[0] == time_dim:
            lons = lons.transpose()
            lats = lats.transpose()

    elif ext == ".csv":
        df = pd.read_csv(data_path)
        df[time_var] = pd.to_datetime(df[time_var])
        start_time = df[time_var].min()
        end_time = df[time_var].max()
        target_time = end_time if elapsed_time is None else start_time + elapsed_time
        df = df[df[time_var] == target_time]

        lons = df[lon_var].values
        lats = df[lat_var].values

    else:
        raise ValueError("Unsupported file format")

    # --- KDE Calculation ---
    kde = gaussian_kde(np.vstack([lons, lats]))
    xi, yi = np.mgrid[min(lons):max(lons):100j, min(lats):max(lats):100j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

    # --- Plotting ---
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    ax.set_extent(extent or [min(lons)-1, max(lons)+1, min(lats)-1, max(lats)+1])

    # Density cloud
    cf = ax.contourf(xi, yi, zi.reshape(xi.shape), levels=20, cmap='viridis', alpha=0.6, transform=ccrs.PlateCarree())
    plt.colorbar(cf, ax=ax, label="Density", shrink=0.8)

    # Optional: plot raw particle positions
    ax.scatter(lons, lats, color="black", s=1, alpha=0.3, transform=ccrs.PlateCarree(), label="Particle positions")

    # Optional: draw node polygons (only one legend entry)
    if show_nodes and node_polys:
        for i, poly in enumerate(node_polys):
            xs, ys = poly.exterior.xy
            label = "Node boundary" if i == 0 else None
            ax.plot(xs, ys, color='red', linewidth=1.2, transform=ccrs.PlateCarree(), label=label)

    # Gridlines and layout
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    # Manually add axis labels
    ax.text(0.5, -0.1, 'Longitude', transform=ax.transAxes,ha='center', va='top', fontsize=12)
    ax.text(-0.08, 0.5, 'Latitude', transform=ax.transAxes,ha='center', va='bottom', rotation='vertical', fontsize=12)

    # Title and legend
    time_label = "Last timestep" if elapsed_time is None else f"t + {elapsed_time}"
    plt.title(f"Particle KDE ({time_label})")
    ax.legend(loc='lower left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Plot saved to {save_path}")
    else:
        plt.show()

def plot_kde_snapshot_with_nodes(
    data_path,
    outputdt,
    elapsed_time=None,
    particles_per_node=100,
    time_var="time",
    time_dim="obs",
    particle_dim="trajectory",
    lon_var="lon",
    lat_var="lat",
    extent=None,
    save_path=None,
    show_nodes=False,
    node_polys=None
):
    """
    Plot node-wise KDE snapshots at a specific time, separating density by origin node.

    This function generates a spatial KDE for each group of particles released
    from the same node, allowing comparison of dispersal footprints by origin.

    Parameters:
    - data_path: Path to dataset (.zarr, .nc, or .csv)
    - outputdt: Timedelta representing model output interval
    - elapsed_time: Time since release to extract snapshot (or None for last timestep)
    - particles_per_node: Number of particles released per node (assumes grouping by order)
    - show_nodes: Whether to draw polygon boundaries for nodes
    - node_polys: List of shapely Polygon geometries
    - extent: Optional [min_lon, max_lon, min_lat, max_lat] for map view
    - save_path: Output filepath for saving plot
    - time_var, time_dim, particle_dim, lon_var, lat_var: Variable/dim names for compatibility

    Notes:
    - Particles are grouped by index for each node's KDE
    - Each node's KDE cloud is drawn with a unique color
    - One node boundary and particle scatter entry are added to the legend
    """
    ext = os.path.splitext(data_path)[-1].lower()

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    
    # Add gridlines with lat/lon labels
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    # Manually add axis labels
    ax.text(0.5, -0.1, 'Longitude', transform=ax.transAxes,ha='center', va='top', fontsize=12)
    ax.text(-0.08, 0.5, 'Latitude', transform=ax.transAxes,ha='center', va='bottom', rotation='vertical', fontsize=12)

    # --- Load and parse data ---
    if ext in [".zarr", ".nc"]:
        ds = xr.open_zarr(data_path) if ext == ".zarr" else xr.open_dataset(data_path)
        num_steps = ds.sizes[time_dim]
        target_index = num_steps - 1 if elapsed_time is None else int(elapsed_time / outputdt)

        lons_all = ds[lon_var].values
        lats_all = ds[lat_var].values

        if ds[lon_var].dims[0] == time_dim:
            lons_all = lons_all.transpose()
            lats_all = lats_all.transpose()

        start_lons = lons_all[:, 0]
        start_lats = lats_all[:, 0]
        curr_lons = lons_all[:, target_index]
        curr_lats = lats_all[:, target_index]

    elif ext == ".csv":
        df = pd.read_csv(data_path)
        df[time_var] = pd.to_datetime(df[time_var])
        start_time = df[time_var].min()
        end_time = df[time_var].max()
        target_time = end_time if elapsed_time is None else start_time + elapsed_time

        df_start = df[df[time_var] == start_time].sort_values("particle_id")
        df_target = df[df[time_var] == target_time].sort_values("particle_id")

        start_lons = df_start[lon_var].values
        start_lats = df_start[lat_var].values
        curr_lons = df_target[lon_var].values
        curr_lats = df_target[lat_var].values

    else:
        raise ValueError("Unsupported file format")

    num_particles = len(curr_lons)
    num_nodes = num_particles // particles_per_node
    cmap = cm.get_cmap("tab10", num_nodes)
    colors = [cmap(i) for i in range(num_nodes)]

    # --- Plot each node group ---
    for i in range(num_nodes):
        start_idx = i * particles_per_node
        end_idx = start_idx + particles_per_node
        node_lons = curr_lons[start_idx:end_idx]
        node_lats = curr_lats[start_idx:end_idx]

        if len(node_lons) == 0:
            continue

        kde = gaussian_kde(np.vstack([node_lons, node_lats]))
        xi, yi = np.mgrid[min(node_lons):max(node_lons):100j, min(node_lats):max(node_lats):100j]
        zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

        # Draw KDE contours
        ax.contour(xi, yi, zi.reshape(xi.shape), levels=10, colors=[colors[i]], linewidths=1, alpha=0.6, transform=ccrs.PlateCarree())

        # Draw particle positions (once, no legend clutter)
        label = "Particles" if i == 0 else None
        ax.scatter(node_lons, node_lats, s=1.5, color=colors[i], alpha=0.3, transform=ccrs.PlateCarree(), label=label)

    # --- Optional: Node polygons ---
    if show_nodes and node_polys:
        for i, poly in enumerate(node_polys):
            xs, ys = poly.exterior.xy
            label = "Node boundary" if i == 0 else None
            ax.plot(xs, ys, color='black', linewidth=1.2, linestyle="--", transform=ccrs.PlateCarree(), label=label)

    # --- Final layout tweaks ---
    ax.set_extent(extent or [min(curr_lons)-1, max(curr_lons)+1, min(curr_lats)-1, max(curr_lats)+1])
    time_label = "Last timestep" if elapsed_time is None else f"t + {elapsed_time}"
    plt.title(f"Node-wise KDE ({time_label})")
       # Add labeled lat/lon gridlines
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    # Manually add axis labels
    ax.text(0.5, -0.1, 'Longitude', transform=ax.transAxes,ha='center', va='top', fontsize=12)
    ax.text(-0.08, 0.5, 'Latitude', transform=ax.transAxes,ha='center', va='bottom', rotation='vertical', fontsize=12)
    ax.legend(loc='lower left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Plot saved to {save_path}")
    else:
        plt.show()

def plot_kde(
    data_path,
    outputdt,
    settlement_hours=0,
    pld_days=None,
    time_var="time",
    time_dim="obs",
    particle_dim="trajectory",
    lon_var="lon",
    lat_var="lat",
    extent=None,
    save_path=None,
    show_nodes=False,
    node_polys=None
):
    """
    Plot a KDE (Kernel Density Estimate) of all particles within their competency window.

    This function visualizes the density of particle positions during the period when
    they are considered competent to settle — from a specified `settlement_hours` after
    release up to a potential PLD (Pelagic Larval Duration) limit.

    Parameters:
    - data_path: Path to input trajectory data (.zarr, .nc, or .csv)
    - outputdt: Time interval between model output steps (as a timedelta)
    - settlement_hours: Minimum particle age (in hours) before competency begins
    - pld_days: Maximum number of days particles are competent; if None, use last timestep
    - time_var, time_dim: Time variable and dimension names (for xarray support)
    - particle_dim: Dimension name for particle ID (for reshaping if needed)
    - lon_var, lat_var: Names of longitude and latitude variables
    - extent: Optional [min_lon, max_lon, min_lat, max_lat] for map boundaries
    - save_path: If provided, saves plot to file instead of displaying
    - show_nodes: Whether to draw overlaid node polygons
    - node_polys: List of shapely Polygons (e.g., node regions or habitats)

    Notes:
    - Flattens all particle positions during the competency window for KDE
    - Adds optional scatter overlay and node outlines
    - Works with long-form CSV or multidimensional xarray datasets
    """
    
    ext = os.path.splitext(data_path)[-1].lower()
    start_index = int(settlement_hours / outputdt.total_seconds() * 3600)

    if ext in [".zarr", ".nc"]:
        ds = xr.open_zarr(data_path) if ext == ".zarr" else xr.open_dataset(data_path)
        num_steps = ds.sizes[time_dim]
        end_index = num_steps - 1 if pld_days is None else min(int(pld_days * 24 / outputdt.total_seconds() * 3600), num_steps - 1)

        lons = ds[lon_var].isel({time_dim: slice(start_index, end_index + 1)}).values
        lats = ds[lat_var].isel({time_dim: slice(start_index, end_index + 1)}).values

        if ds[lon_var].dims[0] == time_dim:
            lons = lons.transpose()
            lats = lats.transpose()

        lons = lons.flatten()
        lats = lats.flatten()

    elif ext == ".csv":
        df = pd.read_csv(data_path)
        df[time_var] = pd.to_datetime(df[time_var])
        start_time = df[time_var].min() + timedelta(hours=settlement_hours)
        end_time = df[time_var].max() if pld_days is None else df[time_var].min() + timedelta(days=pld_days)
        df = df[(df[time_var] >= start_time) & (df[time_var] <= end_time)]

        lons = df[lon_var].values
        lats = df[lat_var].values

    else:
        raise ValueError("Unsupported file format")

    # --- KDE ---
    kde = gaussian_kde(np.vstack([lons, lats]))
    xi, yi = np.mgrid[min(lons):max(lons):100j, min(lats):max(lats):100j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

    # --- Plotting ---
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    ax.set_extent(extent or [min(lons)-1, max(lons)+1, min(lats)-1, max(lats)+1])

    cf = ax.contourf(xi, yi, zi.reshape(xi.shape), levels=20, cmap='viridis', alpha=0.6, transform=ccrs.PlateCarree())
    plt.colorbar(cf, ax=ax, label="Density", shrink=0.8)
    ax.scatter(lons, lats, color="black", s=1, alpha=0.3, transform=ccrs.PlateCarree(), label="Particle positions")

    if show_nodes and node_polys:
        for i, poly in enumerate(node_polys):
            xs, ys = poly.exterior.xy
            label = "Node boundary" if i == 0 else None
            ax.plot(xs, ys, color='red', linewidth=1.2, transform=ccrs.PlateCarree(), label=label)

    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.text(0.5, -0.1, 'Longitude', transform=ax.transAxes,ha='center', va='top', fontsize=12)
    ax.text(-0.08, 0.5, 'Latitude', transform=ax.transAxes,ha='center', va='bottom', rotation='vertical', fontsize=12)

    plt.title("Particle KDE (competency window)")
    ax.legend(loc='lower left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Plot saved to {save_path}")
    else:
        plt.show()

def plot_kde_with_nodes(
    data_path,
    outputdt,
    settlement_hours=0,
    pld_days=None,
    particles_per_node=100,
    time_var="time",
    time_dim="obs",
    particle_dim="trajectory",
    lon_var="lon",
    lat_var="lat",
    extent=None,
    save_path=None,
    show_nodes=False,
    node_polys=None
):
    """
    Plot KDE clouds grouped by particle origin node during the competency window.

    This function computes and visualizes a separate KDE for each release node using
    particle positions from the settlement window (post-settlement threshold through PLD).
    It helps compare dispersal density by source node.

    Parameters:
    - data_path: Path to input data (.zarr, .nc, or .csv)
    - outputdt: Time interval between output steps (timedelta)
    - settlement_hours: Hours post-release when particles become competent to settle
    - pld_days: Maximum duration in days particles remain competent (None = full sim)
    - particles_per_node: Number of particles released per node (used for grouping)
    - time_var, time_dim: Time variable and dimension names
    - particle_dim: Name of the particle ID dimension
    - lon_var, lat_var: Names of longitude and latitude variables
    - extent: Optional map bounds [min_lon, max_lon, min_lat, max_lat]
    - save_path: File path to save the plot (if not displaying)
    - show_nodes: Whether to overlay node polygons
    - node_polys: List of shapely Polygons defining spatial nodes

    Notes:
    - Particles are assumed to be ordered by node in the dataset
    - KDEs are colored uniquely per node; scatter overlays show position spread
    - Only positions during the competency window are included
    """
    
    ext = os.path.splitext(data_path)[-1].lower()
    start_index = int(settlement_hours / outputdt.total_seconds() * 3600)

    if ext in [".zarr", ".nc"]:
        ds = xr.open_zarr(data_path) if ext == ".zarr" else xr.open_dataset(data_path)
        num_steps = ds.sizes[time_dim]
        end_index = num_steps - 1 if pld_days is None else min(int(pld_days * 24 / outputdt.total_seconds() * 3600), num_steps - 1)

        lons_all = ds[lon_var].isel({time_dim: slice(start_index, end_index + 1)}).values
        lats_all = ds[lat_var].isel({time_dim: slice(start_index, end_index + 1)}).values

        if ds[lon_var].dims[0] == time_dim:
            lons_all = lons_all.transpose()
            lats_all = lats_all.transpose()

    elif ext == ".csv":
        df = pd.read_csv(data_path)
        df[time_var] = pd.to_datetime(df[time_var])
        start_time = df[time_var].min() + timedelta(hours=settlement_hours)
        end_time = df[time_var].max() if pld_days is None else df[time_var].min() + timedelta(days=pld_days)
        df = df[(df[time_var] >= start_time) & (df[time_var] <= end_time)].sort_values("particle_id")

        grouped = df.groupby("particle_id")
        lons_all = np.vstack([group[lon_var].values for _, group in grouped])
        lats_all = np.vstack([group[lat_var].values for _, group in grouped])

    else:
        raise ValueError("Unsupported file format")

    num_particles = lons_all.shape[0]
    num_nodes = num_particles // particles_per_node
    cmap = cm.get_cmap("tab10", num_nodes)
    colors = [cmap(i) for i in range(num_nodes)]

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    # Add labeled lat/lon gridlines
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    # Manually add axis labels
    ax.text(0.5, -0.1, 'Longitude', transform=ax.transAxes,ha='center', va='top', fontsize=12)
    ax.text(-0.08, 0.5, 'Latitude', transform=ax.transAxes,ha='center', va='bottom', rotation='vertical', fontsize=12)

    for i in range(num_nodes):
        start_idx = i * particles_per_node
        end_idx = start_idx + particles_per_node
        node_lons = lons_all[start_idx:end_idx].flatten()
        node_lats = lats_all[start_idx:end_idx].flatten()

        if len(node_lons) == 0:
            continue

        kde = gaussian_kde(np.vstack([node_lons, node_lats]))
        xi, yi = np.mgrid[min(node_lons):max(node_lons):100j, min(node_lats):max(node_lats):100j]
        zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

        ax.contour(xi, yi, zi.reshape(xi.shape), levels=10, colors=[colors[i]], linewidths=1, alpha=0.6, transform=ccrs.PlateCarree())
        label = "Particles" if i == 0 else None
        ax.scatter(node_lons, node_lats, s=1.5, color=colors[i], alpha=0.3, transform=ccrs.PlateCarree(), label=label)

    if show_nodes and node_polys:
        for i, poly in enumerate(node_polys):
            xs, ys = poly.exterior.xy
            label = "Node boundary" if i == 0 else None
            ax.plot(xs, ys, color='black', linewidth=1.2, linestyle="--", transform=ccrs.PlateCarree(), label=label)

    ax.set_extent(extent or [np.min(lons_all)-1, np.max(lons_all)+1, np.min(lats_all)-1, np.max(lats_all)+1])
    plt.title("Node-wise KDE (competency window)")
    ax.legend(loc='lower left')
    plt.tight_layout()

    plt.show()


def plot_connectivity_graph_map(
    graph,
    shapefile_path,
    community_attribute="community",
    title="Connectivity Graph",
    node_size=8,
    edge_color="gray",
    edge_alpha=0.5,
    edge_width_scale=1.0,
    node_cmap="tab10",
    show_labels=True,
    extent=None,
    save_path=None,
    show_polygons=True,
    show_arrows=False,
    arrow_size=0.2
):
    """
    Plot a connectivity graph over a geographic map using node polygon centroids.

    Parameters:
    - graph: An igraph.Graph object with node 'name' attributes matching node_id in shapefile
    - shapefile_path: Path to the shapefile with node polygons (must include 'node_id')
    - community_attribute: Node attribute to color by (default = 'community')
    - title: Title for the plot
    - node_size: Size of the node markers
    - edge_color: Color of the edges
    - edge_alpha: Transparency of edges
    - edge_width_scale: Multiply edge weight by this for visible line thickness
    - node_cmap: Colormap name for node communities
    - show_labels: Whether to show node labels
    - extent: Optional map bounds [min_lon, max_lon, min_lat, max_lat]
    - save_path: File path to save the figure instead of showing
    - show_polygons: If True, draw the full node polygons
    - show_arrows: If True and graph is directed, show arrows on edges
    - arrow_size: Arrowhead size (only if show_arrows is True)
    """

    # Load and index shapefile
    gdf = gpd.read_file(shapefile_path)

    # Project to UTM for accurate centroids, then back to WGS84
    utm_crs = gdf.estimate_utm_crs()
    gdf_proj = gdf.to_crs(utm_crs)
    gdf["centroid"] = gdf_proj.centroid.to_crs("EPSG:4326")
    gdf = gdf.set_index("node_id")

    # Build node position dictionary
    node_positions = {
        node: (gdf.loc[node, "centroid"].x, gdf.loc[node, "centroid"].y)
        for node in graph.vs["name"]
    }

    # Setup map
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title(title)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND)
    ax.gridlines(draw_labels=True)

    # Optional: draw the full node polygons
    if show_polygons:
        gdf.reset_index(drop=True).geometry.boundary.plot(
            ax=ax, color="black", linewidth=1, transform=ccrs.PlateCarree())

    # Plot edges
    weights = graph.es["weight"] if "weight" in graph.es.attributes() else [1] * len(graph.es)
    for i, edge in enumerate(graph.es):
        src = graph.vs[edge.source]["name"]
        tgt = graph.vs[edge.target]["name"]
        x0, y0 = node_positions[src]
        x1, y1 = node_positions[tgt]
        lw = weights[i] * edge_width_scale

        if show_arrows and graph.is_directed():
            # Convert geographic to display coordinates
            src_disp = ax.projection.transform_point(x0, y0, ccrs.PlateCarree())
            tgt_disp = ax.projection.transform_point(x1, y1, ccrs.PlateCarree())

            arrow = FancyArrowPatch(
                posA=src_disp,
                posB=tgt_disp,
                arrowstyle='->',
                color=edge_color,
                alpha=edge_alpha,
                linewidth=lw,
                mutation_scale=arrow_size * 100,
                transform=mtransforms.IdentityTransform()
            )
            ax.add_patch(arrow)
        else:
            ax.plot([x0, x1], [y0, y1],
                    color=edge_color,
                    linewidth=lw,
                    alpha=edge_alpha,
                    transform=ccrs.PlateCarree())

    # Plot nodes
    communities = graph.vs[community_attribute]
    cmap = cm.get_cmap(node_cmap)

    for v in graph.vs:
        x, y = node_positions[v["name"]]
        color = cmap(communities[v.index] % cmap.N)
        ax.plot(x, y, marker='o', markersize=node_size, color=color, transform=ccrs.PlateCarree())
        if show_labels:
            ax.text(x, y, str(v["name"]), fontsize=8, ha='center', va='bottom', transform=ccrs.PlateCarree())

    # Adjust map extent
    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Saved graph map to: {save_path}")
    else:
        plt.show()


