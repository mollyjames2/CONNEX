from connex.examples.generate_example_data import generate_connex_data

# Define node centers (longitude, latitude) for the simulation
node_centers = [
    (30.0, -38.0),
    (30.25, -38.5),
    (30.5, -38.0),
    (30.75, -38.5),
    (31.0, -38.0),
]

# Run the data generation function
result = generate_connex_data(
    ocean_data_path_or_url="https://raw.githubusercontent.com/OceanParcels/parcels/master/parcels/examples/GlobCurrent_example_data.nc",
    node_centers=node_centers,
    particles_per_node=100,
    output_dir="data",
    diffusion=100.0,
    runtime_days=3
)

# Print file paths and metadata needed for later analysis
print("\n✅ Example data generated.")
print(f"Output path: {result['output_path']}")
print(f"Node IDs: {result['node_ids']}")
print(f"Particles per node: {result['particles_per_node']}")

