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
from shapely.geometry import Point


def start_node_assignment(
    data_path,
    node_polys,
    node_ids,
    time_var="time",
    time_dim="obs",
    particle_dim="trajectory",
    lon_var="lon",
    lat_var="lat"
):
    """
    Assign each particle to a node based on their initial positions (t=0),
    using actual node names or IDs from shapefile.

    Parameters:
    - data_path: Path to data file (.zarr, .nc, or .csv)
    - node_polys: List of shapely Polygons representing node regions
    - node_ids: List of node names/IDs (same order as node_polys)
    - time_var, time_dim: Time variable and dimension names
    - lon_var, lat_var: Variable names for longitude and latitude

    Returns:
    - assigned_nodes: List of node_id (or None) for each particle
    """
    import os
    import xarray as xr
    import pandas as pd
    from shapely.geometry import Point

    ext = os.path.splitext(data_path)[-1].lower()

    if ext in [".zarr", ".nc"]:
        ds = xr.open_zarr(data_path) if ext == ".zarr" else xr.open_dataset(data_path)

        subset = ds.isel({time_dim: 0})

        lons = subset[lon_var]
        lats = subset[lat_var]

        if lons.ndim == 2 and lons.dims != (particle_dim,):
            lons = lons.transpose(particle_dim)
            lats = lats.transpose(particle_dim)

        lon_start = lons.values
        lat_start = lats.values

    elif ext == ".csv":
        import numpy as np
        df = pd.read_csv(data_path)
        df[time_var] = pd.to_datetime(df[time_var])
        t0 = df[time_var].min()
        df_start = df[df[time_var] == t0].sort_values("particle_id")

        lon_start = df_start[lon_var].values
        lat_start = df_start[lat_var].values

    else:
        raise ValueError("Unsupported file format. Use .zarr, .nc, or .csv")

    # Assign node name/ID for each particle
    assigned_nodes = []
    for lon, lat in zip(lon_start, lat_start):
        pt = Point(lon, lat)
        found = False
        for nid, poly in zip(node_ids, node_polys):
            if poly.contains(pt):
                assigned_nodes.append(nid)
                found = True
                break
        if not found:
            assigned_nodes.append(None)

    return assigned_nodes

    
def plot_trajectories(
    data_path,
    pld=None,
    outputdt=None,
    extent=None,
    show_nodes=False,
    node_polys=None,
    save_path=None,
    time_var="time",
    time_dim="obs",
    particle_dim="trajectory",
    lon_var="lon",
    lat_var="lat",
    color_by_node=False,
    start_nodes=None
):
    """
    Plot particle trajectories from time = 0 up to a specified pelagic larval duration (PLD), 
    using input from NetCDF, Zarr, or CSV files. Optionally colors trajectories by starting node.

    Parameters:
    ----------
    data_path : str
        Path to the input data file (.nc, .zarr, or .csv) containing trajectory information.

    pld : int, optional
        Pelagic larval duration in days. Trajectories are plotted from t=0 to this value. 
        If None, all time steps are included.

    outputdt : timedelta, optional
        Time interval between output steps. Required if `pld` is provided, to calculate time index range.

    extent : list of float, optional
        Map extent in the form [xmin, xmax, ymin, ymax] to limit plot view.

    show_nodes : bool, default False
        Whether to overlay node polygons on the plot.

    node_polys : list of shapely Polygon objects, optional
        Polygons representing spatial node regions (used if `show_nodes=True`).

    save_path : str, optional
        Path to save the generated plot as an image. If None, the plot is displayed interactively.

    time_var : str, default "time"
        Name of the time variable in the dataset.

    time_dim : str, default "obs"
        Name of the time dimension.

    particle_dim : str, default "trajectory"
        Name of the particle/trajectory dimension.

    lon_var : str, default "lon"
        Name of the longitude variable.

    lat_var : str, default "lat"
        Name of the latitude variable.

    color_by_node : bool, default False
        If True, particles are colored based on their `start_nodes` assignment.

    start_nodes : list or array-like, optional
        A list or array of node IDs corresponding to the starting node for each particle. 
        Required if `color_by_node=True`.

    Returns:
    -------
    None
        Displays or saves a plot of particle trajectories.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import xarray as xr
    import pandas as pd
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from datetime import timedelta

    print(f"Reading data from: {data_path}")
    ext = os.path.splitext(data_path)[-1].lower()
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # --- Color setup ---
    if color_by_node:
        if start_nodes is None:
            raise ValueError("start_nodes must be provided if color_by_node is True.")
        unique_nodes = sorted(set(n for n in start_nodes if n is not None))
        cmap = cm.get_cmap("tab10", len(unique_nodes))
        node_color_map = {node: cmap(i) for i, node in enumerate(unique_nodes)}

    # --- Load and filter data ---
    if ext in [".zarr", ".nc"]:
        ds = xr.open_zarr(data_path) if ext == ".zarr" else xr.open_dataset(data_path)
        num_steps = ds.sizes[time_dim]
        time_axis = ds[time_var].dims[0]

        if pld is None:
            time_idxs = np.arange(0, num_steps)
        else:
            if outputdt is None:
                raise ValueError("`outputdt` is required when `pld` is provided.")
            steps_per_day = int(timedelta(days=1) / outputdt)
            target_idx = pld * steps_per_day
            time_idxs = np.arange(0, min(target_idx + 1, num_steps))

        lons = ds[lon_var].isel({time_axis: time_idxs}).values
        lats = ds[lat_var].isel({time_axis: time_idxs}).values

        if ds[lon_var].dims[0] == time_dim:
            lons = lons.transpose()
            lats = lats.transpose()
            
        # Subset over time using isel (by name)
        subset = ds.isel({time_dim: time_idxs})

        # Extract lon/lat: resulting dims might still be [obs, trajectory] or [trajectory, obs]
        lons = subset[lon_var]
        lats = subset[lat_var]

        # Transpose explicitly to 
        if lons.dims != (particle_dim, time_dim):
            lons = lons.transpose(particle_dim, time_dim)
            lats = lats.transpose(particle_dim, time_dim)

        # Convert to numpy
        lons = lons.values
        lats = lats.values

        for i in range(lons.shape[0]):
            color = node_color_map.get(start_nodes[i], "grey") if color_by_node else "blue"
            ax.plot(lons[i], lats[i], linewidth=0.8, alpha=0.6, transform=ccrs.PlateCarree(), color=color)
            ax.plot(lons[i, 0], lats[i, 0], marker='o', color=color, markersize=3, transform=ccrs.PlateCarree())
            ax.plot(lons[i, -1], lats[i, -1], marker='x', color=color, markersize=3, transform=ccrs.PlateCarree())

    elif ext == ".csv":
        df = pd.read_csv(data_path)
        df[time_var] = pd.to_datetime(df[time_var])
        start_time = df[time_var].min()

        if pld is not None:
            if outputdt is None:
                raise ValueError("`outputdt` is required when `pld` is provided.")
            end_time = start_time + timedelta(days=pld)
            df = df[(df[time_var] >= start_time) & (df[time_var] <= end_time)]

        for i, (pid, group) in enumerate(df.groupby("particle_id")):
            color = node_color_map.get(start_nodes[i], "grey") if color_by_node else "blue"
            ax.plot(group[lon_var], group[lat_var], linewidth=0.8, alpha=0.6, transform=ccrs.PlateCarree(), color=color)
            ax.plot(group[lon_var].iloc[0], group[lat_var].iloc[0], marker='o', color=color, markersize=3, transform=ccrs.PlateCarree())
            ax.plot(group[lon_var].iloc[-1], group[lat_var].iloc[-1], marker='x', color=color, markersize=3, transform=ccrs.PlateCarree())

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
    ax.text(0.5, -0.1, 'Longitude', transform=ax.transAxes, ha='center', va='top', fontsize=12)
    ax.text(-0.1, 0.5, 'Latitude', transform=ax.transAxes, ha='center', va='bottom', rotation='vertical', fontsize=12)

    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    plt.title("Particle Trajectories (Colored by Start Node)" if color_by_node else "Particle Trajectories")
    ax.plot([], [], marker='o', color='black', linestyle='None', label='Start')
    ax.plot([], [], marker='x', color='black', linestyle='None', label='End')
    ax.legend(loc='lower left')

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_depth_timeseries(
    data_path,
    depth_var="z",
    time_var="time",
    particle_dim="trajectory",
    time_dim="obs",
    start_nodes=None,
    particle_ids=None,
    color_by_node=False,
    outputdt=None,
    pld=None,
    save_path=None,
    min_depth=0,
    max_depth=None
):
    """
    Plot depth over time for particles from trajectory data.

    Parameters:
    - data_path (str): Path to NetCDF, Zarr, or CSV file.
    - depth_var (str): Name of depth variable (default "z").
    - time_var (str): Name of time variable (default "time").
    - particle_dim (str): Name of particle dimension.
    - time_dim (str): Name of time dimension.
    - start_nodes (list or array): Optional node assignments (used for coloring if color_by_node=True).
    - particle_ids (list of int): Optional list of specific particle indices to plot.
    - color_by_node (bool): Whether to color by start_nodes.
    - outputdt (timedelta): Required if `pld` is specified.
    - pld (int): Pelagic larval duration in days (optional).
    - save_path (str): Path to save the figure (optional).
    - min_depth (float): Minimum depth limit on y-axis (default 0).
    - max_depth (float): Maximum depth limit on y-axis (default None, i.e. auto)
    """
    import os
    import xarray as xr
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from datetime import timedelta

    ext = os.path.splitext(data_path)[-1].lower()
    fig, ax = plt.subplots(figsize=(10, 6))

    if color_by_node:
        if start_nodes is None:
            raise ValueError("start_nodes must be provided when color_by_node is True.")
        unique_nodes = sorted(set(n for n in start_nodes if n is not None))
        cmap = cm.get_cmap("tab10", len(unique_nodes))
        node_color_map = {node: cmap(i) for i, node in enumerate(unique_nodes)}

    if ext in [".zarr", ".nc"]:
        ds = xr.open_zarr(data_path) if ext == ".zarr" else xr.open_dataset(data_path)
        num_steps = ds.sizes[time_dim]

        if pld is not None:
            if outputdt is None:
                raise ValueError("`outputdt` must be provided if `pld` is used.")
            max_idx = min(int(timedelta(days=pld) / outputdt) + 1, num_steps)
            ds = ds.isel({time_dim: slice(0, max_idx)})

        # Handle time variable
        if particle_dim in ds[time_var].dims:
            times = pd.to_datetime(ds[time_var].isel({particle_dim: 0}).values)
        else:
            times = pd.to_datetime(ds[time_var].values)

        depths = ds[depth_var]

        # Ensure dimensions exist
        required_dims = {particle_dim, time_dim}
        if not required_dims.issubset(set(depths.dims)):
            raise ValueError(f"Depth variable must have dimensions {particle_dim} and {time_dim}, got {depths.dims}")

        depths = depths.transpose(particle_dim, time_dim)

        # Auto-flip depth if needed
        median_depth = depths.median().item()
        if median_depth < 0:
            print("Detected depth is positive-up; converting to positive-down.")
            depths = -depths

        if particle_ids is None:
            particle_ids = range(depths.shape[0])  # Plot all particles

        for i in particle_ids:
            if i >= depths.shape[0]:
                print(f"Skipping particle {i}: index out of bounds.")
                continue

            if color_by_node:
                node = start_nodes[i]
                color = node_color_map.get(node, "black")
                # Add one legend entry per node
                existing_labels = ax.get_legend_handles_labels()[1]
                label = f"Node {node}" if f"Node {node}" not in existing_labels else None
            else:
                color = "blue"
                label = None

            ax.plot(times, depths[i, :], color=color, alpha=0.7, label=label)

    elif ext == ".csv":
        df = pd.read_csv(data_path)

        # Auto-flip depth if needed
        median_depth = df[depth_var].median()
        if median_depth < 0:
            print("Detected depth is positive-up; converting to positive-down.")
            df[depth_var] = -df[depth_var]

        df[time_var] = pd.to_datetime(df[time_var])

        if pld is not None:
            start_time = df[time_var].min()
            end_time = start_time + timedelta(days=pld)
            df = df[df[time_var] <= end_time]

        if "particle_id" not in df.columns:
            raise ValueError("CSV must contain a 'particle_id' column.")

        if particle_ids is None:
            particle_ids = df["particle_id"].unique()[:10]

        for i, pid in enumerate(particle_ids):
            sub = df[df["particle_id"] == pid]
            if sub.empty:
                print(f"Warning: No data for particle {pid}")
                continue

            if color_by_node:
                node = start_nodes[i]
                color = node_color_map.get(node, "black")
                existing_labels = ax.get_legend_handles_labels()[1]
                label = f"Node {node}" if f"Node {node}" not in existing_labels else None
            else:
                color = "blue"
                label = None

            ax.plot(sub[time_var], sub[depth_var], color=color, alpha=0.7, label=label)

    else:
        raise ValueError("Unsupported file format")

    ax.invert_yaxis()
    ax.set_title("Particle Depth Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Depth (m)")
    ax.grid(True)

    if min_depth is not None or max_depth is not None:
        ax.set_ylim(bottom=max_depth, top=min_depth)  # reversed for inverted axis

    if color_by_node:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize="small")

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_kde_snapshot(
    data_path,
    outputdt,
    pld=None,
    time_var="time",
    time_dim="obs",
    particle_dim="trajectory",
    lon_var="lon",
    lat_var="lat",
    extent=None,
    save_path=None,
    show_nodes=False,
    node_polys=None,
    show_particles=True,
    kde_resolution=100,
    bandwidth=None 
):
    """
    Plot a KDE (kernel density estimate) cloud for all particles at a specific timestep,
    based on PLD (pelagic larval duration) in days.
    
    Parameters:
    - data_path (str): Path to input data (NetCDF, Zarr, or CSV) containing particle trajectories.
    - outputdt (timedelta): Timestep interval of the data (used to convert PLD to a timestep index).
    - pld (int or None): Pelagic larval duration in days. Determines the timestep to extract a position snapshot.
                        If None, uses the final timestep.
    - time_var (str): Name of the time variable in the dataset (default is "time").
    - time_dim (str): Name of the time dimension in the dataset (default is "obs").
    - particle_dim (str): Name of the particle/trajectory dimension (default is "trajectory").
    - lon_var (str): Name of the longitude variable (default is "lon").
    - lat_var (str): Name of the latitude variable (default is "lat").
    - extent (list or tuple): Optional [min_lon, max_lon, min_lat, max_lat] to set map extent.
    - save_path (str or None): If specified, saves the plot to this path (e.g., PNG). Otherwise, displays the plot.
    - show_nodes (bool): Whether to plot node polygons on top of the KDE.
    - node_polys (list of shapely.Polygon): Optional list of polygons (e.g., source nodes) to overlay on the plot.
    - show_particles (bool): Whether to also plot particle positions as scatter points at the selected timestep.
    - kde_resolution (int or tuple): Grid resolution for the KDE (evaluated on an N×N grid). Higher = smoother but slower.
    - bandwidth (float or str, optional): Controls KDE smoothing. Smaller = finer detail; larger = smoother density.
        - float: scales the bandwidth (e.g. 0.3 = more detailed)
        - 'scott' or 'silverman': use standard KDE rules
        - None (default): same as 'scott'
    """

    ext = os.path.splitext(data_path)[-1].lower()

    if ext in [".zarr", ".nc"]:
        ds = xr.open_zarr(data_path) if ext == ".zarr" else xr.open_dataset(data_path)
        num_steps = ds.sizes[time_dim]
        target_index = num_steps - 1 if pld is None else int(timedelta(days=pld) / outputdt)
        target_index = min(target_index, num_steps - 1)

        lons = ds[lon_var].isel({time_dim: target_index}).values
        lats = ds[lat_var].isel({time_dim: target_index}).values

        if ds[lon_var].dims[0] == time_dim:
            lons = lons.transpose()
            lats = lats.transpose()

    elif ext == ".csv":
        df = pd.read_csv(data_path)
        df[time_var] = pd.to_datetime(df[time_var])
        start_time = df[time_var].min()
        target_time = df[time_var].max() if pld is None else start_time + timedelta(days=pld)
        df = df[df[time_var] == target_time]

        lons = df[lon_var].values
        lats = df[lat_var].values

    else:
        raise ValueError("Unsupported file format")

    # KDE
       # --- KDE grid resolution ---
    if isinstance(kde_resolution, int):
        nx, ny = kde_resolution, kde_resolution
    elif isinstance(kde_resolution, (tuple, list)) and len(kde_resolution) == 2:
        nx, ny = kde_resolution
    else:
        raise ValueError("kde_resolution must be an int or a tuple of (nx, ny)")

    kde = gaussian_kde(np.vstack([lons, lats]),bw_method=bandwidth)
    xi, yi = np.mgrid[
        min(lons):max(lons):complex(nx),
        min(lats):max(lats):complex(ny)
    ]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    ax.set_extent(extent or [min(lons)-1, max(lons)+1, min(lats)-1, max(lats)+1])
    cf = ax.contourf(xi, yi, zi.reshape(xi.shape), levels=20, cmap='viridis', alpha=0.6, transform=ccrs.PlateCarree())
    plt.colorbar(cf, ax=ax, label="Density", shrink=0.8)
    
    if show_particles:
        ax.scatter(lons, lats, color="black", s=1, alpha=0.3, transform=ccrs.PlateCarree(), label="Particle positions")

    if show_nodes and node_polys:
        for i, poly in enumerate(node_polys):
            xs, ys = poly.exterior.xy
            label = "Node boundary" if i == 0 else None
            ax.plot(xs, ys, color='red', linewidth=1.2, transform=ccrs.PlateCarree(), label=label)

    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.text(0.5, -0.1, 'Longitude', transform=ax.transAxes, ha='center', va='top', fontsize=12)
    ax.text(-0.1, 0.5, 'Latitude', transform=ax.transAxes, ha='center', va='bottom', rotation='vertical', fontsize=12)

    time_label = "Last timestep" if pld is None else f"{pld} days"
    plt.title(f"Particle KDE ({time_label})")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc='lower left')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f" Plot saved to {save_path}")
    else:
        plt.show()
        
    
def plot_kde_snapshot_with_nodes(
    data_path,
    outputdt,
    pld=None,
    start_nodes=None,
    node_ids=None,
    time_var="time",
    time_dim="obs",
    particle_dim="trajectory",
    lon_var="lon",
    lat_var="lat",
    extent=None,
    save_path=None,
    show_nodes=False,
    node_polys=None,
    show_particles=True,
    kde_resolution=100,
    bandwidth=None
):
    """
    Plot node-wise KDE (Kernel Density Estimate) snapshots at a specific timestep 
    determined by the pelagic larval duration (PLD), grouping particles by their 
    actual starting node IDs.

    Parameters:
    - data_path (str): Path to input trajectory data file (NetCDF, Zarr, or CSV).
    - outputdt (timedelta): Timestep interval of the dataset; used to convert PLD (in days) to an index.
    - pld (int or None): Pelagic larval duration in days. If provided, selects the timestep corresponding to this duration.
                         If None, the last available timestep is used.
    - start_nodes (list or array): Node index assigned to each particle at t=0. Required for grouping particles by node.
    - node_ids (list of int or None): Optional subset of node IDs to include in the plot. If None, all nodes present in
                                      `start_nodes` are used.
    - time_var (str): Name of the time variable in the dataset (default: "time").
    - time_dim (str): Name of the time dimension (default: "obs").
    - particle_dim (str): Name of the particle/trajectory dimension (default: "trajectory").
    - lon_var (str): Name of the longitude variable (default: "lon").
    - lat_var (str): Name of the latitude variable (default: "lat").
    - extent (list or tuple of float, optional): Geographic map extent [min_lon, max_lon, min_lat, max_lat].
    - save_path (str or None): If specified, saves the output figure to this path (e.g., PNG). Otherwise, the plot is shown.
    - show_nodes (bool): Whether to overlay node polygons on the map (requires `node_polys`).
    - node_polys (list of shapely.Polygon or None): Optional list of polygons representing node boundaries to plot.
    - show_particles (bool): Whether to also plot the raw particle positions as scatter points at the selected timestep.
    - kde_resolution (int or tuple of two ints): Grid resolution used for KDE computation:
        - If int: uses a square grid (e.g., 100×100).
        - If tuple: uses a rectangular grid (nx, ny).
        Higher values yield smoother, more detailed density maps but increase computational cost.
    - bandwidth (float or str, optional): Controls KDE smoothing. Smaller = finer detail; larger = smoother density.
        - float: scales the bandwidth (e.g. 0.3 = more detailed)
        - 'scott' or 'silverman': use standard KDE rules
        - None (default): same as 'scott'


    Returns:
    - None. Displays or saves a KDE plot grouped by starting node.
    """

 

    ext = os.path.splitext(data_path)[-1].lower()

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.text(0.5, -0.1, 'Longitude', transform=ax.transAxes, ha='center', va='top', fontsize=12)
    ax.text(-0.1, 0.5, 'Latitude', transform=ax.transAxes, ha='center', va='bottom', rotation='vertical', fontsize=12)

    if ext in [".zarr", ".nc"]:
        ds = xr.open_zarr(data_path) if ext == ".zarr" else xr.open_dataset(data_path)
        num_steps = ds.sizes[time_dim]
        target_index = num_steps - 1 if pld is None else int(timedelta(days=pld) / outputdt)
        target_index = min(target_index, num_steps - 1)

        lons_all = ds[lon_var].values
        lats_all = ds[lat_var].values

        if ds[lon_var].dims[0] == time_dim:
            lons_all = lons_all.transpose()
            lats_all = lats_all.transpose()

        curr_lons = lons_all[:, target_index]
        curr_lats = lats_all[:, target_index]

    elif ext == ".csv":
        df = pd.read_csv(data_path)
        df[time_var] = pd.to_datetime(df[time_var])
        start_time = df[time_var].min()
        target_time = df[time_var].max() if pld is None else start_time + timedelta(days=pld)

        df_target = df[df[time_var] == target_time].sort_values("particle_id")

        curr_lons = df_target[lon_var].values
        curr_lats = df_target[lat_var].values

    else:
        raise ValueError("Unsupported file format")

    if start_nodes is None:
        raise ValueError("You must provide start_nodes (list of node_id for each particle).")

    unique_nodes = sorted(set(n for n in start_nodes if n is not None))
    node_to_indices = {nid: [] for nid in unique_nodes}
    for i, nid in enumerate(start_nodes):
        if nid is not None:
            node_to_indices[nid].append(i)

    cmap = cm.get_cmap("tab10", len(unique_nodes))
    colors = {nid: cmap(i) for i, nid in enumerate(unique_nodes)}

    for nid, indices in node_to_indices.items():
        node_lons = curr_lons[indices]
        node_lats = curr_lats[indices]

        if len(node_lons) == 0:
            continue

           # --- KDE grid resolution ---
        if isinstance(kde_resolution, int):
            nx, ny = kde_resolution, kde_resolution
        elif isinstance(kde_resolution, (tuple, list)) and len(kde_resolution) == 2:
            nx, ny = kde_resolution
        else:
            raise ValueError("kde_resolution must be an int or a tuple of (nx, ny)")

        kde = gaussian_kde(np.vstack([lons, lats]),bw_method=bandwidth)
        xi, yi = np.mgrid[
            min(lons):max(lons):complex(nx),
            min(lats):max(lats):complex(ny)
        ]
        zi = kde(np.vstack([xi.flatten(), yi.flatten()]))


        ax.contour(xi, yi, zi.reshape(xi.shape), levels=10, colors=[colors[nid]], linewidths=1, alpha=0.6, transform=ccrs.PlateCarree())
        
        if show_particles:
            ax.scatter(node_lons, node_lats, s=1.5, color=colors[nid], alpha=0.3, transform=ccrs.PlateCarree(), label=f"Node {nid}")

    if show_nodes and node_polys:
        for i, poly in enumerate(node_polys):
            xs, ys = poly.exterior.xy
            label = "Node boundary" if i == 0 else None
            ax.plot(xs, ys, color='black', linewidth=1.2, linestyle="--", transform=ccrs.PlateCarree(), label=label)

    ax.set_extent(extent or [min(curr_lons)-1, max(curr_lons)+1, min(curr_lats)-1, max(curr_lats)+1])
    time_label = "Last timestep" if pld is None else f"{pld} days"
    plt.title(f"Node-wise KDE ({time_label})")
    ax.legend(loc='lower left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f" Plot saved to {save_path}")
    else:
        plt.show()



def plot_kde(
    data_path,
    outputdt,
    settlement_hours=0,
    pld_days=None,
    timestep=None,
    time_var="time",
    time_dim="obs",
    particle_dim="trajectory",
    lon_var="lon",
    lat_var="lat",
    extent=None,
    save_path=None,
    show_nodes=False,
    node_polys=None,
    show_particles=True,
    kde_resolution=100,
    bandwidth=None 
):
    """
    Plot a kernel density estimate (KDE) of particle positions during their competency window.

    Parameters:
    - data_path (str): Path to the input file (.zarr, .nc, or .csv).
    - outputdt (timedelta): Time interval between model output steps.
    - settlement_hours (int): Minimum age in hours before particles are competent to settle.
    - pld_days (int or None): Pelagic larval duration in days. If None, includes up to the final timestep.
    - timestep (timedelta or None): Optional sampling interval. Must be ≥ outputdt.
    - time_var (str): Name of the time variable in the dataset (default: "time").
    - time_dim (str): Name of the time dimension (default: "obs").
    - particle_dim (str): Name of the particle/trajectory dimension (default: "trajectory").
    - lon_var (str): Name of the longitude variable (default: "lon").
    - lat_var (str): Name of the latitude variable (default: "lat").
    - extent (list or tuple): Optional map bounds [min_lon, max_lon, min_lat, max_lat].
    - save_path (str or None): If provided, saves the plot to the specified file path.
    - show_nodes (bool): If True, overlays polygons (e.g., settlement nodes).
    - node_polys (list or None): Optional list of Shapely polygons to plot if show_nodes is True.
    - show_particles (bool): Whether to also plot the individual particle positions.
    - kde_resolution (int or tuple): Grid resolution for KDE. Can be an integer or a (nx, ny) tuple.
        - Higher values = finer KDE mesh (more detailed but slower).
    - bandwidth (float, str, or None): Bandwidth for KDE smoothing.
        - float: scaling factor (e.g., 0.3 = narrower bandwidth, more detail).
        - 'scott' or 'silverman': standard KDE rules.
        - None: defaults to 'scott'.

    Returns:
    - None. Displays or saves a 2D KDE plot of particle density during their competency window.
    """

    import os
    import numpy as np
    import pandas as pd
    import xarray as xr
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from datetime import timedelta
    from scipy.stats import gaussian_kde

    ext = os.path.splitext(data_path)[-1].lower()

    if timestep is not None:
        if timestep < outputdt:
            raise ValueError(f"`timestep` ({timestep}) must be >= `outputdt` ({outputdt})")
        step_factor = int(timestep / outputdt)
    else:
        step_factor = 1

    start_index = int(settlement_hours / outputdt.total_seconds() * 3600)

    if ext in [".zarr", ".nc"]:
        ds = xr.open_zarr(data_path) if ext == ".zarr" else xr.open_dataset(data_path)
        num_steps = ds.sizes[time_dim]
        end_index = num_steps - 1 if pld_days is None else min(
            int(pld_days * 24 / outputdt.total_seconds() * 3600), num_steps - 1
        )

        selected = ds.isel({time_dim: slice(start_index, end_index + 1, step_factor)})
        lons = selected[lon_var].values
        lats = selected[lat_var].values

        if selected[lon_var].dims[0] == time_dim:
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

        if timestep is not None:
            df["_rounded_time"] = (df[time_var] - pd.Timestamp("1970-01-01")) // timestep
            df = df.drop_duplicates(subset=["_rounded_time", "particle_id"])
            df = df.drop(columns=["_rounded_time"])

        lons = df[lon_var].values
        lats = df[lat_var].values

    else:
        raise ValueError("Unsupported file format")

    # --- KDE grid resolution ---
    if isinstance(kde_resolution, int):
        nx, ny = kde_resolution, kde_resolution
    elif isinstance(kde_resolution, (tuple, list)) and len(kde_resolution) == 2:
        nx, ny = kde_resolution
    else:
        raise ValueError("kde_resolution must be an int or a tuple of (nx, ny)")

    kde = gaussian_kde(np.vstack([lons, lats]),bw_method=bandwidth)
    xi, yi = np.mgrid[
        min(lons):max(lons):complex(nx),
        min(lats):max(lats):complex(ny)
    ]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

    # --- Plotting ---
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    ax.set_extent(extent or [min(lons)-1, max(lons)+1, min(lats)-1, max(lats)+1])

    cf = ax.contourf(xi, yi, zi.reshape(xi.shape), levels=20, cmap='viridis', alpha=0.6, transform=ccrs.PlateCarree())
    plt.colorbar(cf, ax=ax, label="Density", shrink=0.8)

    if show_particles:
        ax.scatter(lons, lats, color="black", s=1, alpha=0.3, transform=ccrs.PlateCarree(), label="Particle positions")

    if show_nodes and node_polys:
        for i, poly in enumerate(node_polys):
            xs, ys = poly.exterior.xy
            label = "Node boundary" if i == 0 else None
            ax.plot(xs, ys, color='red', linewidth=1.2, transform=ccrs.PlateCarree(), label=label)

    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.text(0.5, -0.1, 'Longitude', transform=ax.transAxes, ha='center', va='top', fontsize=12)
    ax.text(-0.1, 0.5, 'Latitude', transform=ax.transAxes, ha='center', va='bottom', rotation='vertical', fontsize=12)

    plt.title("Particle KDE (competency window)")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
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
    start_nodes=None,
    time_var="time",
    time_dim="obs",
    particle_dim="trajectory",
    lon_var="lon",
    lat_var="lat",
    extent=None,
    save_path=None,
    show_nodes=False,
    node_polys=None,
    show_particles=True,
    kde_resolution=100,
    bandwidth=None
):
    """
    Plot KDE (Kernel Density Estimate) clouds of particle positions grouped by their origin nodes
    during the competency window (from settlement_hours to pld_days).

    Particles are grouped and colored based on their start_nodes assignment, allowing visualization
    of spatial distributions per source node.

    Parameters:
    - data_path (str): Path to particle trajectory data file (.zarr, .nc, or .csv).
    - outputdt (timedelta): Time interval between model output steps.
    - settlement_hours (int, default=0): Minimum particle age in hours before competency starts.
    - pld_days (int or None, default=None): Pelagic larval duration in days; if None, uses data up to last timestep.
    - start_nodes (list or array-like): List mapping each particle to its origin node (required if grouping by node).
    - time_var (str, default="time"): Name of the time variable in dataset.
    - time_dim (str, default="obs"): Name of the time dimension.
    - particle_dim (str, default="trajectory"): Name of the particle dimension.
    - lon_var (str, default="lon"): Name of the longitude variable.
    - lat_var (str, default="lat"): Name of the latitude variable.
    - extent (list or tuple of floats or None): Optional map extent [min_lon, max_lon, min_lat, max_lat].
    - save_path (str or None): Path to save the plot image; if None, plot is shown interactively.
    - show_nodes (bool, default=False): Whether to overlay node boundary polygons on the plot.
    - node_polys (list of shapely.geometry.Polygon or None): List of polygons defining node boundaries.
    - show_particles (bool, default=True): Whether to overlay individual particle positions on the plot.
    - kde_resolution (int or tuple of int, default=100): Number of grid points for KDE in each spatial dimension.
      Can be a single integer or a tuple (nx, ny).
    - bandwidth (float, str, or None, default=None): Bandwidth for KDE smoothing.
      Can be a float scaling factor, or 'scott', 'silverman', or None (defaults to 'scott').

    Returns:
    - None. Displays or saves a KDE plot visualizing particle density per origin node.
    """


    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from scipy.stats import gaussian_kde
    from matplotlib import cm

    ext = os.path.splitext(data_path)[-1].lower()
    start_index = int(settlement_hours / outputdt.total_seconds() * 3600)

    if ext in [".zarr", ".nc"]:
        ds = xr.open_zarr(data_path) if ext == ".zarr" else xr.open_dataset(data_path)
        num_steps = ds.sizes[time_dim]
        end_index = num_steps - 1 if pld_days is None else min(
            int(pld_days * 24 / outputdt.total_seconds() * 3600), num_steps - 1
        )

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

    if start_nodes is None:
        raise ValueError("start_nodes must be provided and match number of particles")

    start_nodes = np.array(start_nodes)
    num_particles = lons_all.shape[0]
    if len(start_nodes) != num_particles:
        raise ValueError("Length of start_nodes does not match number of particles")

    unique_nodes = sorted(np.unique(start_nodes))
    cmap = cm.get_cmap("tab10", len(unique_nodes))
    node_to_color = {node: cmap(i) for i, node in enumerate(unique_nodes)}

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.LAND)

    # Gridlines
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    ax.text(0.5, -0.1, 'Longitude', transform=ax.transAxes, ha='center', va='top', fontsize=12)
    ax.text(-0.1, 0.5, 'Latitude', transform=ax.transAxes, ha='center', va='bottom', rotation='vertical', fontsize=12)

    for node_id in unique_nodes:
        particle_idxs = np.where(start_nodes == node_id)[0]
        node_lons = lons_all[particle_idxs].flatten()
        node_lats = lats_all[particle_idxs].flatten()

        if len(node_lons) == 0:
            continue
        
        
        # --- KDE grid resolution ---
        if isinstance(kde_resolution, int):
            nx, ny = kde_resolution, kde_resolution
        elif isinstance(kde_resolution, (tuple, list)) and len(kde_resolution) == 2:
            nx, ny = kde_resolution
        else:
            raise ValueError("kde_resolution must be an int or a tuple of (nx, ny)")

        kde = gaussian_kde(np.vstack([lons, lats]),bw_method=bandwidth)
        xi, yi = np.mgrid[
            min(lons):max(lons):complex(nx),
            min(lats):max(lats):complex(ny)
        ]
        zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

        color = node_to_color[node_id]
        ax.contour(xi, yi, zi.reshape(xi.shape), levels=10, colors=[color], linewidths=1, alpha=0.6, transform=ccrs.PlateCarree())
        
        if show_particles:
            ax.scatter(node_lons, node_lats, s=1.5, color=color, alpha=0.3, transform=ccrs.PlateCarree(), label=f"Node {node_id}")

    if show_nodes and node_polys:
        for i, poly in enumerate(node_polys):
            xs, ys = poly.exterior.xy
            label = "Node boundary" if i == 0 else None
            ax.plot(xs, ys, color='black', linewidth=1.2, linestyle="--", transform=ccrs.PlateCarree(), label=label)

    ax.set_extent(extent or [
        np.min(lons_all) - 1, np.max(lons_all) + 1,
        np.min(lats_all) - 1, np.max(lats_all) + 1
    ])
    plt.title("Node-wise KDE (competency window)")
    ax.legend(loc='lower left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
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


