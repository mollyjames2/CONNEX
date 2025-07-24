import os
import numpy as np
import geopandas as gpd
import xarray as xr
from shapely.geometry import box
from datetime import timedelta
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionDiffusionM1
from parcels import download_example_dataset
import glob

"""
generate_example_data.py

This module generates example larval connectivity data for the CONNEX tool.

Steps:
1. Downloads or loads ocean current data
2. Defines spatial polygon nodes
3. Releases particles in each polygon
4. Simulates trajectories using Parcels
5. Outputs:
    - A trajectory Zarr dataset
    - A shapefile of polygons

Use `generate_connex_data()` to run the full workflow.
"""

# Author: Molly James
# Year: 2025
# Part of: connex.examples.generate_example_data

def download_ocean_data(_, __):
    """Downloads and returns the folder path to the GlobCurrent example NetCDF dataset."""
    print("Downloading GlobCurrent example dataset via Parcels...")
    dataset_dir = download_example_dataset("GlobCurrent_example_data")

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    print(f"Dataset directory: {dataset_dir}")
    return dataset_dir


def create_polygons(node_centers, half_size=0.125, output_path="data/node_shp.shp"):
    """Creates square polygons centered on node coordinates and saves as shapefile."""
    node_polys = [box(lon - half_size, lat - half_size, lon + half_size, lat + half_size)
                  for lon, lat in node_centers]
    node_ids = list(range(1, len(node_polys) + 1))

    gdf = gpd.GeoDataFrame({'node_id': node_ids}, geometry=node_polys, crs="EPSG:4326")
    gdf.to_file(output_path)
    print(f"Polygon shapefile saved to: {output_path}")
    return node_ids, node_polys


def generate_particles(node_polys, particles_per_node):
    """Generates random particles within each polygon."""
    all_lon, all_lat = [], []
    for poly in node_polys:
        minx, miny, maxx, maxy = poly.bounds
        lons = np.random.uniform(minx, maxx, particles_per_node)
        lats = np.random.uniform(miny, maxy, particles_per_node)
        all_lon.extend(lons)
        all_lat.extend(lats)
    return all_lon, all_lat


def run_simulation(dataset_dir, lon, lat, output_path, diffusion=100.0,runtime_days=10, dt=timedelta(minutes=10),outputdt=timedelta(hours=6)):
    """Runs Parcels simulation with advection and diffusion and saves output."""
    filenames = {
        "U": f"{dataset_dir}/20*.nc",
        "V": f"{dataset_dir}/20*.nc",
    }
    variables = {'U': 'eastward_eulerian_current_velocity', 'V': 'northward_eulerian_current_velocity'}
    dimensions = {'lat': 'lat', 'lon': 'lon', 'time': 'time'}

    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions)
    fieldset.add_constant_field("Kh_zonal", diffusion, mesh="spherical")
    fieldset.add_constant_field("Kh_meridional", diffusion, mesh="spherical")
    fieldset.add_constant('dres', 0.00005)  # required for AdvectionRK4DiffusionM1

    pset = ParticleSet.from_list(fieldset=fieldset, pclass=JITParticle, lon=lon, lat=lat)

    kernels = AdvectionDiffusionM1

    pset.execute(
        kernels,
        runtime=timedelta(days=runtime_days),
        dt=dt,
        output_file=pset.ParticleFile(name=output_path,outputdt=outputdt)
    )

    print(f"Simulation complete. Output saved to {output_path}")


def generate_connex_data(
    ocean_data_path_or_url,
    node_centers,
    particles_per_node=100,
    output_dir="data",
    diffusion=10.0,
    runtime_days=3,
    dt=timedelta(minutes=10),
    outputdt=timedelta(hours=6)
):
    """Main entrypoint for generating simulation data."""
    os.makedirs(output_dir, exist_ok=True)

    dataset_dir = ocean_data_path_or_url
    if ocean_data_path_or_url.startswith("http"):
        dataset_dir = download_ocean_data(
            ocean_data_path_or_url,
            os.path.join(output_dir, "ocean_data.nc")
        )

    shape_path = os.path.join(output_dir, "node_shp.shp")
    node_ids, node_polys = create_polygons(node_centers, output_path=shape_path)

    lons, lats = generate_particles(node_polys, particles_per_node)

    output_path = os.path.join(output_dir, "example_trajectories.zarr")
    run_simulation(dataset_dir, lons, lats, output_path, diffusion=diffusion,runtime_days=runtime_days,dt = dt, outputdt = outputdt)

    return {
        "output_path": output_path,
        "node_ids": node_ids,
        "node_polys": node_polys,
        "particles_per_node": particles_per_node
    }
