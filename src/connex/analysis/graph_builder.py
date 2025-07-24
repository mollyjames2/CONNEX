import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np
from shapely.geometry import Point
import os
from datetime import timedelta

def build_connectivity_matrix_start_end(
    data_path,
    shapefile_path,
    pld_days=None,
    time_var="time",
    time_dim='obs',
    particle_dim="trajectory"
):
    """
    Builds a node-to-node connectivity matrix from start and end positions,
    where the end time is defined by PLD in days from the release.
    """
    gdf = gpd.read_file(shapefile_path)
    node_ids = gdf["node_id"].tolist()
    node_polys = gdf.geometry
    spatial_index = node_polys.sindex
    matrix = pd.DataFrame(0, index=node_ids, columns=node_ids)

    ext = os.path.splitext(data_path)[-1].lower()
    ds = (
        xr.open_zarr(data_path) if ext == ".zarr" or os.path.isdir(data_path)
        else xr.open_dataset(data_path) if ext in [".nc", ".netcdf"]
        else None
    )
    if ds is None:
        raise ValueError("Only .zarr or .nc formats are supported.")

    times = pd.to_datetime(ds[time_var].isel({particle_dim: 0}).values)
    release_time = times[0]

    if pld_days is not None:
        target_time = release_time + timedelta(days=pld_days)
        t_end_idx = int(np.searchsorted(times, target_time, side="right")) - 1
        t_end_idx = min(t_end_idx, len(times) - 1)
    else:
        t_end_idx = len(times) - 1

    lon_start = ds.lon.isel({time_dim: 0}).values
    lat_start = ds.lat.isel({time_dim: 0}).values
    lon_end = ds.lon.isel({time_dim: t_end_idx}).values
    lat_end = ds.lat.isel({time_dim: t_end_idx}).values

    start_points = gpd.GeoSeries([Point(xy) for xy in zip(lon_start, lat_start)], crs="EPSG:4326")
    end_points = gpd.GeoSeries([Point(xy) for xy in zip(lon_end, lat_end)], crs="EPSG:4326")

    def get_node_ids(points):
        node_ids_out = []
        for pt in points:
            possible = list(spatial_index.intersection(pt.bounds))
            found = None
            for idx in possible:
                if node_polys.iloc[idx].contains(pt):
                    found = gdf["node_id"].iloc[idx]
                    break
            node_ids_out.append(found)
        return node_ids_out

    sources = get_node_ids(start_points)
    targets = get_node_ids(end_points)

    for src, tgt in zip(sources, targets):
        if src is not None and tgt is not None:
            matrix.at[src, tgt] += 1

    print(f" Matrix built: {len(sources)} particles evaluated")
    return matrix

def build_connectivity_matrix_by_path(
    data_path,
    shapefile_path,
    settlement_hours=None,
    pld_days=None,
    outputdt=timedelta(hours=1),
    time_var="time",
    time_dim="obs",
    particle_dim="trajectory",
    lon_var="lon",
    lat_var="lat"
):
    """
    Build a path-based connectivity matrix by counting whether each particle
    passes through each node polygon within the competency window.
    """
    gdf = gpd.read_file(shapefile_path)
    node_ids = gdf["node_id"].tolist()
    node_polys = gdf.geometry.tolist()
    spatial_index = gdf.sindex
    matrix = pd.DataFrame(0, index=node_ids, columns=node_ids)

    ext = os.path.splitext(data_path)[-1].lower()
    ds = (
        xr.open_zarr(data_path) if ext == ".zarr" or os.path.isdir(data_path)
        else xr.open_dataset(data_path) if ext in [".nc", ".netcdf"]
        else None
    )
    if ds is None:
        raise ValueError("Only .zarr or .nc formats are supported.")

    times = pd.to_datetime(ds[time_var].isel({particle_dim: 0}).values)
    release_time = times[0]

    settle_start_time = release_time + timedelta(hours=settlement_hours) if settlement_hours else release_time
    start_idx = int(np.searchsorted(times, settle_start_time, side="left"))
    end_idx = len(times) - 1

    if pld_days is not None:
        end_time = release_time + timedelta(days=pld_days)
        end_idx = int(np.searchsorted(times, end_time, side="right")) - 1
        end_idx = min(end_idx, len(times) - 1)

    if start_idx >= len(times):
        raise ValueError("Settlement window starts after trajectory ends.")

    valid_idxs = np.arange(start_idx, end_idx + 1)

    print(f" Competency window: from {times[start_idx]} (t={start_idx}) to {times[end_idx]} (t={end_idx})")

    lon = ds[lon_var].isel({time_dim: valid_idxs}).values
    lat = ds[lat_var].isel({time_dim: valid_idxs}).values

    if ds[lon_var].dims[0] == time_dim:
        lon = lon.T
        lat = lat.T

    n_particles = lon.shape[0]
    particles_per_node = n_particles // len(node_ids)

    for i, src_id in enumerate(node_ids):
        start_idx_p = i * particles_per_node
        end_idx_p = (i + 1) * particles_per_node

        src_lon = lon[start_idx_p:end_idx_p]
        src_lat = lat[start_idx_p:end_idx_p]

        for p in range(src_lon.shape[0]):
            visited = set()
            for t in range(src_lon.shape[1]):
                pt = Point(src_lon[p, t], src_lat[p, t])
                for idx in spatial_index.intersection(pt.bounds):
                    if node_polys[idx].contains(pt):
                        node_id = node_ids[idx]
                        visited.add(node_id)
                        break
            for target_id in visited:
                matrix.at[src_id, target_id] += 1

    print(f"? Path-based connectivity matrix built from {n_particles} particles.")
    return matrix
