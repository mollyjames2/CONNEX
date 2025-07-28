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




def summarize_connectivity_start_end(
    ds,
    start_nodes,
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
        start_nodes (list[int or None]): List of node_id for each particle (same length as number of particles).
        node_ids (list[int]): Unique node IDs matching node_polys.
        node_polys (list[Polygon]): Shapely Polygons defining each node.
        lon_var (str): Longitude variable name.
        lat_var (str): Latitude variable name.
        time_var (str): Time variable.
        time_dim (str): Time dimension.
        particle_dim (str): Particle dimension.
        pld_days (int or None): Pelagic larval duration. If None, use last timestep.

    Returns:
        dict: Dictionary with per-node summary (retained, to_others, lost, total).
    """
    # Extract time axis and find end time index
    times = pd.to_datetime(ds[time_var].isel({particle_dim: 0}).values)
    release_time = times[0]

    if pld_days is not None:
        target_time = release_time + timedelta(days=pld_days)
        end_idx = int(np.searchsorted(times, target_time, side="right")) - 1
        end_idx = min(end_idx, len(times) - 1)
    else:
        end_idx = len(times) - 1

    print(f" Release time: {release_time}")
    print(f" End time (PLD = {pld_days} days): {times[end_idx]} (index {end_idx})")

    # Extract end positions
    lon_end = ds[lon_var].isel({time_dim: end_idx}).values
    lat_end = ds[lat_var].isel({time_dim: end_idx}).values

    if ds[lon_var].dims[0] == time_dim:
        lon_end = lon_end.T
        lat_end = lat_end.T

    # Map node_id to polygon for lookup
    node_poly_map = dict(zip(node_ids, node_polys))

    # Initialize summary
    summary = {nid: {
        "retained": 0,
        "to_others": defaultdict(int),
        "lost": 0,
        "total": 0
    } for nid in node_ids}

    # Iterate over particles
    for i, source_node in enumerate(start_nodes):
        if source_node is None:
            continue  # skip unassigned particles

        summary[source_node]["total"] += 1
        pt = Point(lon_end[i], lat_end[i])

        found = False
        for target_node, poly in node_poly_map.items():
            if poly.contains(pt):
                if target_node == source_node:
                    summary[source_node]["retained"] += 1
                else:
                    summary[source_node]["to_others"][target_node] += 1
                found = True
                break

        if not found:
            summary[source_node]["lost"] += 1

    # Print summary
    for nid in node_ids:
        s = summary[nid]
        total = s["total"]
        if total == 0:
            continue
        print(f"\n Node {nid}:")
        print(f"   Retained: {s['retained']} ({s['retained'] / total:.1%})")
        for tid, count in s["to_others"].items():
            print(f"   To node {tid}: {count} ({count / total:.1%})")
        print(f"   Lost: {s['lost']} ({s['lost'] / total:.1%})")

    print("\n Start-end connectivity summary complete.")
    return summary




def summarize_connectivity_by_path(
    ds,
    start_nodes,
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
    and BEFORE a maximum duration (pld_days), using start_nodes instead of fixed particles per node.

    Parameters:
        ds (xarray.Dataset): Trajectory dataset
        start_nodes (list): Node ID (from node_ids) assigned to each particle at release
        node_ids (list): List of all node IDs
        node_polys (list): Shapely polygons for each node
        settlement_hours (int): Min age (in hours) before particles can settle
        pld_days (int or None): Max time (in days) particles remain competent
        outputdt (timedelta): Output interval of model
        lon_var, lat_var (str): Coordinate variable names
        time_var (str): Name of the time coordinate
        time_dim (str): Name of the time dimension
        particle_dim (str): Name of the particle dimension

    Returns:
        dict: Summary of retained, to_others, lost, total per source node
    """
    import numpy as np
    import pandas as pd
    from shapely.geometry import Point

    # Validate competency window
    if pld_days is not None:
        max_settlement_hours = pld_days * 24
        if settlement_hours > max_settlement_hours:
            raise ValueError(
                f"Invalid configuration: settlement_hours ({settlement_hours}h) "
                f"is longer than PLD duration ({pld_days} days = {max_settlement_hours}h)."
            )

    # Extract time axis
    times = pd.to_datetime(ds[time_var].isel({particle_dim: 0}).values)
    t0 = times[0]
    settle_start_time = t0 + timedelta(hours=settlement_hours)

    start_idx = int(np.searchsorted(times, settle_start_time, side="left"))
    end_idx = len(times) - 1
    if pld_days is not None:
        max_time = t0 + timedelta(days=pld_days)
        end_idx = int(np.searchsorted(times, max_time, side="right")) - 1
        end_idx = min(end_idx, len(times) - 1)

    if start_idx >= len(times):
        raise ValueError("Settlement window starts after trajectory ends.")

    print(f"⏱ Competency window: from {times[start_idx]} (t={start_idx}) to {times[end_idx]} (t={end_idx})")

    valid_idxs = np.arange(start_idx, end_idx + 1)
    lon_data = ds[lon_var].isel({time_dim: valid_idxs}).values
    lat_data = ds[lat_var].isel({time_dim: valid_idxs}).values

    if ds[lon_var].dims[0] == time_dim:
        lon_data = lon_data.T
        lat_data = lat_data.T

    n_particles = lon_data.shape[0]
    start_nodes = np.array(start_nodes)
    summary = {}

    for source_id in node_ids:
        particle_idxs = np.where(start_nodes == source_id)[0]

        retained = 0
        lost = 0
        connected_to = {tid: 0 for tid in node_ids if tid != source_id}

        for p in particle_idxs:
            visited = set()
            for t in range(lon_data.shape[1]):
                pt = Point(lon_data[p, t], lat_data[p, t])
                for tid, poly in zip(node_ids, node_polys):
                    if poly.contains(pt):
                        visited.add(tid)

            final_lon = ds[lon_var].isel({time_dim: end_idx, particle_dim: p}).values
            final_lat = ds[lat_var].isel({time_dim: end_idx, particle_dim: p}).values
            final_pt = Point(float(final_lon), float(final_lat))
            in_start_node_at_end = node_polys[node_ids.index(source_id)].contains(final_pt)

            if visited == {source_id} and in_start_node_at_end:
                retained += 1
            else:
                for tid in visited:
                    if tid != source_id:
                        connected_to[tid] += 1
                if not visited or (visited == {source_id} and not in_start_node_at_end):
                    lost += 1

        total = len(particle_idxs)
        summary[source_id] = {
            "retained": retained,
            "to_others": connected_to,
            "lost": lost,
            "total": total
        }

        print(f"\n Node {source_id}:")
        print(f"   Retained: {retained} ({retained / total:.1%})")
        for tid, count in connected_to.items():
            print(f"   Reached node {tid}: {count} ({count / total:.1%})")
        print(f"   Lost: {lost} ({lost / total:.1%})")

    print("\n✅ Path-based connectivity summary complete.")
    return summary


