"""
generate_data.py

This script generates example input data for the CONNEX tool 
(Connectivity Of Nodes and NEtwork eXploration). It performs the following tasks:

1. Downloads a sample ocean current dataset (GlobCurrent)
2. Defines 3 spatial polygon nodes as 0.5° x 0.5° boxes
3. Releases 100 particles per node, randomly within each polygon
4. Runs a 2-day Parcels simulation using advection + horizontal diffusion (100 m²/s)
5. Outputs:
   - A NetCDF file of larval trajectories (data/example_trajectories.nc)
   - A shapefile of polygon nodes (data/node_shp.*)
6. Reports how many particles from each node:
   - Were retained
   - Reached another node
   - Were lost outside the network
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box, Point
from datetime import timedelta
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, BrownianMotion2D

# -------------------------
# Setup
# -------------------------
os.makedirs("data", exist_ok=True)

# -------------------------
# Step 1: Download ocean data
# -------------------------
print("Checking for ocean current data...")

globcurrent_url = "https://raw.githubusercontent.com/OceanParcels/parcels/master/parcels/examples/GlobCurrent_example_data.nc"
globcurrent_file = "data/GlobCurrent_example_data.nc"

if not os.path.exists(globcurrent_file):
    print("Downloading GlobCurrent example data...")
    os.system(f"wget -O {globcurrent_file} {globcurrent_url}")
else:
    print("GlobCurrent file already exists.")

# -------------------------
# Step 2: Create 3 polygon nodes
# -------------------------
print("Creating polygon node shapefile...")

# Centers of 3 nodes (lon, lat)
centers = [(31.0, -38.0), (31.5, -38.0), (32.0, -38.0)]
node_polys = []
node_ids = []

half_size = 0.25  # 0.5° x 0.5° box

for i, (lon_c, lat_c) in enumerate(centers, 1):
    poly = box(lon_c - half_size, lat_c - half_size,
               lon_c + half_size, lat_c + half_size)
    node_polys.append(poly)
    node_ids.append(i)

gdf = gpd.GeoDataFrame({'node_id': node_ids}, geometry=node_polys, crs="EPSG:4326")
gdf.to_file("data/node_shp.shp")

print("Shapefile saved to data/node_shp.*")

# -------------------------
# Step 3: Generate particle release points
# -------------------------
print("Generating 300 particles (100 per node)...")

n_particles_per_node = 100
n_nodes = len(node_ids)
all_lon, all_lat = [], []

for poly in node_polys:
    minx, miny, maxx, maxy = poly.bounds
    lons = np.random.uniform(minx, maxx, n_particles_per_node)
    lats = np.random.uniform(miny, maxy, n_particles_per_node)
    all_lon.extend(lons)
    all_lat.extend(lats)

print(f"Total particles: {len(all_lon)}")

# -------------------------
# Step 4: Run Parcels simulation
# -------------------------
print("Running Parcels simulation with horizontal diffusivity...")

filenames = {'U': globcurrent_file, 'V': globcurrent_file}
variables = {'U': 'eastward_eulerian_current_velocity', 'V': 'northward_eulerian_current_velocity'}
dimensions = {'lat': 'lat', 'lon': 'lon', 'time': 'time'}

fieldset = FieldSet.from_netcdf(filenames, variables, dimensions)

# Add constant horizontal diffusion
fieldset.add_constant_field("Kh_zonal", 100, mesh="spherical")
fieldset.add_constant_field("Kh_meridional", 100, mesh="spherical")

# Create particles
pset = ParticleSet.from_list(fieldset=fieldset,
                             pclass=JITParticle,
                             lon=all_lon,
                             lat=all_lat)

# Run the simulation
output_path = "data/example_trajectories"
kernel = AdvectionRK4 + BrownianMotion2D

pset.execute(kernel,
             runtime=timedelta(days=2),
             dt=timedelta(minutes=30),
             output_file=pset.ParticleFile(name=output_path, outputdt=timedelta(hours=6)))

print("Parcels trajectory output saved to NetCDF.")

# -------------------------
# Step 5: Analyze arrivals by node
# -------------------------
print("\nAnalyzing final particle destinations...\n")

# Load final particle positions
ds = xr.open_dataset("data/example_trajectories.nc")
final_lons = ds.lon[-1, :].values
final_lats = ds.lat[-1, :].values

for i, source_id in enumerate(node_ids):
    start_idx = i * n_particles_per_node
    end_idx = (i + 1) * n_particles_per_node

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

    total = n_particles_per_node
    print(f"Node {source_id}:")
    print(f"  Retained: {retained} ({retained / total:.1%})")
    for target_id in to_others:
        count = to_others[target_id]
        print(f"  To node {target_id}: {count} ({count / total:.1%})")
    print(f"  Lost outside network: {lost} ({lost / total:.1%})\n")

print("✅ Connectivity summary complete.")

