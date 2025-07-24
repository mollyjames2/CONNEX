import os
import xarray as xr
import pandas as pd
import numpy as np
from shapely.geometry import Point
from datetime import timedelta
from collections import defaultdict


def open_trajectory_data(path, time_var="time"):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".zarr" or os.path.isdir(path):
        return xr.open_zarr(path)
    elif ext in [".nc", ".netcdf"]:
        return xr.open_dataset(path)
    elif ext == ".csv":
        return pd.read_csv(path, parse_dates=[time_var])
    else:
        raise ValueError(f"Unsupported file type: {ext}")


import pandas as pd
import numpy as np
from shapely.geometry import Point
from datetime import timedelta

from shapely.geometry import Point
from datetime import timedelta
import numpy as np
import pandas as pd

def summarize_connectivity_start_end(
    ds,
    particles_per_node,
    node_ids,
    node_polys,
    lon_var="lon",
    lat_var="lat",
    time_var="time",
    time_dim="obs",
    particle_dim="trajectory",
    pld_days: int = None
):
    """
    Summarize connectivity between nodes based on particle start and end positions.

    Parameters:
        ds (xarray.Dataset): Trajectory dataset.
        particles_per_node (int): Number of particles released per node.
        node_ids (list): List of node IDs.
        node_polys (list): List of Shapely Polygons for each node.
        lon_var (str): Longitude variable name.
        lat_var (str): Latitude variable name.
        time_var (str): Time coordinate variable name.
        time_dim (str): Name of the time dimension in dataset.
        particle_dim (str): Name of the particle dimension in dataset.
        pld_days (int or None): Pelagic Larval Duration in days. If None, use last timestep.

    Returns:
        dict: Summary for each node including retained, to_others, lost, and total particles.
    """
    # Extract 1D time axis by picking from one particle
    times = pd.to_datetime(ds[time_var].isel({particle_dim: 0}).values)
    release_time = times[0]

    # Compute end time index based on PLD
    if pld_days is not None:
        target_time = release_time + timedelta(days=pld_days)
        end_idx = int(np.searchsorted(times, target_time, side="right")) - 1
        end_idx = min(end_idx, len(times) - 1)
    else:
        end_idx = len(times) - 1

    print(f" Release time: {release_time}")
    print(f" End time (PLD = {pld_days} days): {times[end_idx]} (index {end_idx})")

    # Extract start and end positions
    lon_start = ds[lon_var].isel({time_dim: 0}).values
    lat_start = ds[lat_var].isel({time_dim: 0}).values
    lon_end = ds[lon_var].isel({time_dim: end_idx}).values
    lat_end = ds[lat_var].isel({time_dim: end_idx}).values

    # Transpose if needed (make shape [particles, timesteps])
    if ds[lon_var].dims[0] == time_dim:
        lon_start, lat_start = lon_start.T, lat_start.T
        lon_end, lat_end = lon_end.T, lat_end.T

    summary = {}

    for i, source_id in enumerate(node_ids):
        part_start = i * particles_per_node
        part_end = (i + 1) * particles_per_node

        retained = 0
        lost = 0
        to_others = {tid: 0 for tid in node_ids if tid != source_id}

        for lon, lat in zip(lon_end[part_start:part_end], lat_end[part_start:part_end]):
            pt = Point(lon, lat)
            assigned = False
            for tid, poly in zip(node_ids, node_polys):
                if poly.contains(pt):
                    assigned = True
                    if tid == source_id:
                        retained += 1
                    else:
                        to_others[tid] += 1
                    break
            if not assigned:
                lost += 1

        summary[source_id] = {
            "retained": retained,
            "to_others": to_others,
            "lost": lost,
            "total": part_end - part_start
        }

        print(f"\n Node {source_id}:")
        print(f"   Retained: {retained} ({retained / particles_per_node:.1%})")
        for tid, count in to_others.items():
            print(f"   To node {tid}: {count} ({count / particles_per_node:.1%})")
        print(f"   Lost: {lost} ({lost / particles_per_node:.1%})")

    print("\n✅ Start-end connectivity summary complete.")
    return summary



def summarize_connectivity_by_path(
    ds,
    particles_per_node,
    node_ids,
    node_polys,
    settlement_hours=0,
    pld_days=None,
    outputdt=timedelta(hours=1),
    lon_var="lon",
    lat_var="lat",
    time_var="time",
    time_dim="obs",
    particle_dim="trajectory"
):
    """
    Summarize connectivity by tracing particles that enter any node AFTER a minimum age (settlement_hours)
    and BEFORE a maximum duration (pld_days).

    Parameters:
        ds (xarray.Dataset): Trajectory dataset
        particles_per_node (int): Particles released per node
        node_ids (list): List of node IDs
        node_polys (list): Shapely polygons for each node
        settlement_hours (int): Min age (in hours) before particles can settle
        pld_days (int or None): Max time (in days) particles remain competent. None means no upper limit.
        outputdt (timedelta): Output interval of model
        lon_var, lat_var (str): Coordinate variable names
        time_var (str): Name of the time coordinate
        time_dim (str): Name of the time dimension
        particle_dim (str): Name of the particle dimension

    Returns:
        dict: Summary of retained, to_others, lost, total per source node
    """
        # Validate that the competency window is not negative
    if pld_days is not None:
        max_settlement_hours = pld_days * 24
        if settlement_hours > max_settlement_hours:
            raise ValueError(
                f"Invalid configuration: settlement_hours ({settlement_hours}h) "
                f"is longer than the PLD duration ({pld_days} days = {max_settlement_hours}h)."
            )

    # Extract 1D time array for indexing
    times = pd.to_datetime(ds[time_var].isel({particle_dim: 0}).values)
    t0 = times[0]
    settle_start_time = t0 + timedelta(hours=settlement_hours)

    # Compute index range
    start_idx = int(np.searchsorted(times, settle_start_time, side="left"))
    end_idx = len(times) - 1

    if pld_days is not None:
        max_time = t0 + timedelta(days=pld_days)
        end_idx = int(np.searchsorted(times, max_time, side="right")) - 1
        end_idx = min(end_idx, len(times) - 1)

    if start_idx >= len(times):
        raise ValueError("Settlement window starts after trajectory ends.")

    print(f"⏱ Competency window: from {times[start_idx]} (t={start_idx}) to {times[end_idx]} (t={end_idx})")

    # Subset data
    valid_idxs = np.arange(start_idx, end_idx + 1)
    lon_data = ds[lon_var].isel({time_dim: valid_idxs}).values
    lat_data = ds[lat_var].isel({time_dim: valid_idxs}).values

    # Ensure shape [particles, timesteps]
    if ds[lon_var].dims[0] == time_dim:
        lon_data = lon_data.T
        lat_data = lat_data.T

    n_particles = lon_data.shape[0]
    summary = {}

    for i, source_id in enumerate(node_ids):
        start_idx_p = i * particles_per_node
        end_idx_p = (i + 1) * particles_per_node

        retained = 0
        lost = 0
        connected_to = {tid: 0 for tid in node_ids if tid != source_id}

        for p in range(start_idx_p, end_idx_p):
            visited = set()
            for t in range(lon_data.shape[1]):
                pt = Point(lon_data[p, t], lat_data[p, t])
                for tid, poly in zip(node_ids, node_polys):
                    if poly.contains(pt):
                        visited.add(tid)

            final_lon = ds[lon_var].isel({time_dim: end_idx, particle_dim: p}).values
            final_lat = ds[lat_var].isel({time_dim: end_idx, particle_dim: p}).values
            final_pt = Point(float(final_lon), float(final_lat))
            
            in_start_node_at_end = node_polys[i].contains(final_pt)

            if visited == {source_id} and in_start_node_at_end:
                retained += 1
            else:
                for tid in visited:
                    if tid != source_id:
                        connected_to[tid] += 1
                if not visited or (visited == {source_id} and not in_start_node_at_end):
                    lost += 1

        summary[source_id] = {
            "retained": retained,
            "to_others": connected_to,
            "lost": lost,
            "total": end_idx_p - start_idx_p
        }
        print(f"\n Node {source_id}:")
        print(f"   Retained: {retained} ({retained / particles_per_node:.1%})")
        for tid, count in connected_to.items():
            print(f"   Reached node {tid}: {count} ({count / particles_per_node:.1%})")
        print(f"   Lost: {lost} ({lost / particles_per_node:.1%})")

    print("\n✅ Path-based connectivity summary complete.")
    return summary

