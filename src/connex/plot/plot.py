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
    ax.gridlines(draw_labels=True)

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
    ax.legend(loc='lower left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Plot saved to {save_path}")
    else:
        plt.show()



