import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np
from shapely.geometry import Point
import os

def build_connectivity_matrix(
    data_path,
    shapefile_path,
    start_time=None,
    end_time=None,
    time_var="time",
    particle_var="trajectory"
):
    """
    Builds a node-to-node connectivity matrix from large trajectory datasets.

    """

    # --- Load node polygons ---
    gdf = gpd.read_file(shapefile_path)
    node_ids = gdf["node_id"].tolist()
    node_polys = gdf.geometry
    spatial_index = node_polys.sindex  # For fast spatial queries

    matrix = pd.DataFrame(0, index=node_ids, columns=node_ids)

    ext = os.path.splitext(data_path)[-1].lower()

    if ext in [".zarr", ".nc"]:
        ds = xr.open_zarr(data_path) if ext == ".zarr" else xr.open_dataset(data_path)

        # Get time values and indices
        times = pd.to_datetime(ds[time_var].values)
        t_start = pd.to_datetime(start_time) if start_time else times[0]
        t_end = pd.to_datetime(end_time) if end_time else times[-1]

        t_start_idx = int(np.argmin(np.abs(times - t_start)))
        t_end_idx = int(np.argmin(np.abs(times - t_end)))

        if t_start_idx == t_end_idx:
            raise ValueError("Start and end time map to the same timestep.")

        # Efficiently select only start and end positions
        lon_start = ds.lon.isel({time_var: t_start_idx}).values
        lat_start = ds.lat.isel({time_var: t_start_idx}).values
        lon_end = ds.lon.isel({time_var: t_end_idx}).values
        lat_end = ds.lat.isel({time_var: t_end_idx}).values

        start_points = gpd.GeoSeries([Point(xy) for xy in zip(lon_start, lat_start)], crs="EPSG:4326")
        end_points = gpd.GeoSeries([Point(xy) for xy in zip(lon_end, lat_end)], crs="EPSG:4326")

    elif ext == ".csv":
        df = pd.read_csv(data_path, usecols=["lon", "lat", "particle_id", "time"])
        df["time"] = pd.to_datetime(df["time"])

        if start_time:
            df = df[df["time"] >= pd.to_datetime(start_time)]
        if end_time:
            df = df[df["time"] <= pd.to_datetime(end_time)]

        grouped = df.groupby("particle_id")
        start_points = gpd.GeoSeries([Point(g.iloc[0]["lon"], g.iloc[0]["lat"]) for _, g in grouped], crs="EPSG:4326")
        end_points = gpd.GeoSeries([Point(g.iloc[-1]["lon"], g.iloc[-1]["lat"]) for _, g in grouped], crs="EPSG:4326")

    else:
        raise ValueError("Unsupported file format. Use .zarr, .nc, or .csv")

    # --- Assign particles to nodes using spatial join ---
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

