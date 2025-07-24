import os
import xarray as xr
import pandas as pd
from shapely.geometry import Point
from datetime import datetime
import numpy as np
from collections import defaultdict

def open_trajectory_data(path, lon_var="lon", lat_var="lat", time_var="time"):
    """Open trajectory data from Zarr, NetCDF, or CSV into a common format."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".zarr" or os.path.isdir(path):
        ds = xr.open_zarr(path)
    elif ext in [".nc", ".netcdf"]:
        ds = xr.open_dataset(path)
    elif ext == ".csv":
        df = pd.read_csv(path, parse_dates=[time_var])
        return df
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return ds


def summarize_connectivity_start_end(
    ds,
    particles_per_node,
    node_ids,
    node_polys,
    lon_var="lon",
    lat_var="lat",
    time_dim="time",
    particle_dim="particle",
    start_time=None,
    end_time=None
):
    """
    Summarize particle retention, transfer, and loss based on final position.

    Args:
        ds (xarray.Dataset or pd.DataFrame): Trajectory dataset.
        particles_per_node (int): Number of particles released per node.
        node_ids (list): List of node IDs.
        node_polys (list): List of Shapely polygons for nodes.
        lon_var (str): Longitude variable name.
        lat_var (str): Latitude variable name.
        time_dim (str): Time dimension name (xarray).
        particle_dim (str): Particle dimension name (xarray).
        start_time (str or datetime): Optional filter start time.
        end_time (str or datetime): Optional filter end time.

    Returns:
        dict: Summary of retained, to_others, lost, total per node.
    """
    if isinstance(ds, pd.DataFrame):
        df = ds.copy()
        if start_time:
            df = df[df["time"] >= pd.to_datetime(start_time)]
        if end_time:
            df = df[df["time"] <= pd.to_datetime(end_time)]

        final_df = df.sort_values(by="time").groupby("particle").last()
        final_lons = final_df[lon_var].values
        final_lats = final_df[lat_var].values

    else:
        if start_time or end_time:
            t = pd.to_datetime(ds[time_dim].values)
            if start_time:
                ds = ds.sel({time_dim: t >= pd.to_datetime(start_time)})
            if end_time:
                ds = ds.sel({time_dim: t <= pd.to_datetime(end_time)})

        try:
            lon_data = ds[lon_var]
            lat_data = ds[lat_var]

            if time_dim in lon_data.dims and particle_dim in lon_data.dims:
                final_lons = lon_data.isel({time_dim: -1}).values
                final_lats = lat_data.isel({time_dim: -1}).values
            else:
                raise ValueError("Expected 2D coordinates with dimensions [time, particle]")

            final_lons = final_lons.flatten()
            final_lats = final_lats.flatten()

        except Exception as e:
            raise RuntimeError(f"Failed to extract final positions: {e}")

    summary = {}

    for i, source_id in enumerate(node_ids):
        start_idx = i * particles_per_node
        end_idx = (i + 1) * particles_per_node

        retained = 0
        to_others = {target_id: 0 for target_id in node_ids if target_id != source_id}
        lost = 0

        for lon, lat in zip(final_lons[start_idx:end_idx], final_lats[start_idx:end_idx]):
            point = Point(lon, lat)
            found = False
            for target_id, poly in zip(node_ids, node_polys):
                if poly.contains(point):
                    found = True
                    if target_id == source_id:
                        retained += 1
                    else:
                        to_others[target_id] += 1
                    break
            if not found:
                lost += 1

        summary[source_id] = {
            "retained": retained,
            "to_others": to_others,
            "lost": lost,
            "total": end_idx - start_idx
        }

        # Optional: print result
        print(f"\nNode {source_id}:")
        print(f"  Retained: {retained} ({retained / particles_per_node:.1%})")
        for target_id, count in to_others.items():
            print(f"  To node {target_id}: {count} ({count / particles_per_node:.1%})")
        print(f"  Lost: {lost} ({lost / particles_per_node:.1%})")

    print("\n✅ Connectivity summary complete.")
    return summary



from shapely.geometry import Point
import pandas as pd
import numpy as np

def summarize_connectivity_by_path(
    ds,
    particles_per_node,
    node_ids,
    node_polys,
    lon_var="lon",
    lat_var="lat",
    time_dim="time",
    particle_dim="particle",
    settlement_hours=None
):
    """
    Summarize particle connectivity by path: track if particles pass through any node during simulation.

    Args:
        ds (xarray.Dataset): Trajectory dataset.
        particles_per_node (int): Number of particles released per node.
        node_ids (list): List of node IDs.
        node_polys (list): List of Shapely polygons for nodes.
        lon_var (str): Longitude variable name.
        lat_var (str): Latitude variable name.
        time_dim (str): Time dimension name.
        particle_dim (str): Particle dimension name.
        settlement_hours (int, optional): Max hours since release during which settlement is counted.

    Returns:
        dict: Summary of retained, to_others, lost, total per source node.
    """
    times = pd.to_datetime(ds[time_dim].values)
    time_deltas = ((times - times[0]) / np.timedelta64(1, "s")) / 3600  # hours as float

    valid_times = (time_deltas <= settlement_hours) if settlement_hours is not None else np.full_like(time_deltas, True, dtype=bool)

    lon_data = ds[lon_var].values  # [particles, timesteps]
    lat_data = ds[lat_var].values

    n_particles = lon_data.shape[0]
    summary = {}

    for i, source_id in enumerate(node_ids):
        start_idx = i * particles_per_node
        end_idx = (i + 1) * particles_per_node

        connected_to = {target_id: 0 for target_id in node_ids if target_id != source_id}
        retained = 0
        lost = 0

        for p in range(start_idx, end_idx):
            visited = set()

            # Check path over time
            for t_idx, valid in enumerate(valid_times):
                if not valid:
                    continue
                point = Point(lon_data[p, t_idx], lat_data[p, t_idx])
                for target_id, poly in zip(node_ids, node_polys):
                    if poly.contains(point):
                        visited.add(target_id)
                        # no break here: we want to detect multiple node visits!

            final_point = Point(lon_data[p, -1], lat_data[p, -1])
            in_start_node_at_end = node_polys[i].contains(final_point)

            if visited == {source_id} and in_start_node_at_end:
                retained += 1
            else:
                for target_id in visited:
                    if target_id != source_id:
                        connected_to[target_id] += 1

                if not visited or (visited == {source_id} and not in_start_node_at_end):
                    lost += 1

        summary[source_id] = {
            "retained": retained,
            "to_others": connected_to,
            "lost": lost,
            "total": end_idx - start_idx
        }

        # Print output
        print(f"\nNode {source_id}:")
        print(f"  Retained: {retained} ({retained / particles_per_node:.1%})")
        for target_id, count in connected_to.items():
            print(f"  Reached node {target_id}: {count} ({count / particles_per_node:.1%})")
        print(f"  Lost: {lost} ({lost / particles_per_node:.1%})")

    print("\n✅ Path-based connectivity summary complete.")
    return summary
