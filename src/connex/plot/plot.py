import os
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.cm as cm

def plot_trajectories(
    data_path,
    extent=None,
    show_nodes=False,
    node_polys=None,
    save_path=None,
    start_time=None,
    end_time=None,
    time_var="time",
    particle_var="trajectory",
    color_by_node=False,
    particles_per_node=None
):
    """
    Plot particle trajectories from .zarr, .nc, or .csv, optionally colored by source node
    and with start/end markers.

    Parameters:
        data_path (str): Path to the dataset file (.zarr, .nc, .csv).
        extent (list): [lon_min, lon_max, lat_min, lat_max].
        show_nodes (bool): Whether to show polygonal node boundaries.
        node_polys (list): List of shapely Polygon objects.
        save_path (str): If set, saves figure to file instead of displaying.
        start_time (str): Start time filter (inclusive).
        end_time (str): End time filter (inclusive).
        time_var (str): Time variable name.
        particle_var (str): Particle dimension name.
        color_by_node (bool): Whether to color trajectories by release node.
        particles_per_node (int): Number of particles released per node.
    """
    ext = os.path.splitext(data_path)[-1].lower()
    print(f" Reading data from: {data_path}")

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Initialize color palette
    colors = None
    if color_by_node and particles_per_node:
        if ext in [".zarr", ".nc"]:
            ds_temp = xr.open_zarr(data_path) if ext == ".zarr" else xr.open_dataset(data_path)
            num_particles = ds_temp.sizes[particle_var]
        elif ext == ".csv":
            df_temp = pd.read_csv(data_path)
            num_particles = df_temp["particle_id"].nunique()
        else:
            raise ValueError("Unsupported file format for coloring.")

        num_nodes = num_particles // particles_per_node
        cmap = cm.get_cmap("tab10", num_nodes)
        colors = [cmap(i) for i in range(num_nodes)]

    if ext in [".zarr", ".nc"]:
        ds = xr.open_zarr(data_path) if ext == ".zarr" else xr.open_dataset(data_path)
        times = pd.to_datetime(ds[time_var].values)

        # Filter by time
        time_mask = np.full(times.shape, True)
        if start_time:
            time_mask &= times >= pd.to_datetime(start_time)
        if end_time:
            time_mask &= times <= pd.to_datetime(end_time)

        time_idxs = np.where(time_mask)[0]
        if len(time_idxs) == 0:
            raise ValueError("No data found in specified time range.")

        lons = ds.lon.isel({ds[time_var].dims[0]: time_idxs}).values
        lats = ds.lat.isel({ds[time_var].dims[0]: time_idxs}).values

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

    # Optional: plot node polygons
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

    # Legend
    ax.plot([], [], marker='o', color='black', linestyle='None', label='Start')
    ax.plot([], [], marker='x', color='black', linestyle='None', label='End')
    ax.legend(loc='lower left')

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Plot saved to {save_path}")
    else:
        plt.show()
