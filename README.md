# CONNEX  
### Connectivity Of Nodes and NEtwork eXploration

A graph-based analysis tool for evaluating marine larval dispersal and ecological connectivity using Lagrangian model outputs.

---

## Overview

**CONNEX** is a modular toolkit that transforms larval dispersal data into directed, weighted networks to quantify and visualize connectivity across marine ecosystems. It is designed to work with output from particle tracking simulations (e.g., Parcels, Ichthyop, etc.) and allows users to analyze how larvae move between predefined spatial units.

By applying network science techniques, CONNEX enables researchers and marine planners to:

- Identify stepping-stone habitats and connectivity hotspots
- Quantify isolation, centrality, and reachability of habitat patches
- Detect communities of well-connected regions using graph theory
- Compare connectivity across species, seasons, or management scenarios
- Support marine spatial planning, conservation design, and metapopulation modeling

---

## How It Works

CONNEX takes larval dispersal outputs in **NetCDF, .Zarr or CSV format**, containing particle tracking data with release and settlement coordinates (and optionally timestamps). It maps these trajectories onto **spatial nodes**, which are defined by a shapefile. These nodes represent areas such as habitat patches, grid cells, or management zones.

You can:

- Provide your own shapefile of polygons to define node boundaries
- Use the built-in node-generation functions to:
  - Create a regular grid of nodes across your study region
  - Generate nodes by clustering initial particle release locations

This flexibility makes it easy to tailor the spatial resolution and structure of your connectivity analysis.

---

## Key Features

- Accepts larval dispersal outputs in **NetCDF, .Zarr or CSV** format  
- Converts trajectories into adjacency matrices or edge lists  
- Builds directed, weighted graphs using `networkx`  
- Calculates metrics like degree, betweenness, clustering, and modularity  
- Detects graph communities using algorithms like Louvain or Girvan–Newman  
- Visualizes connectivity networks with spatial overlays
- plots dispersal trajectories and dispersal density clouds  
- Filters and thresholds links based on frequency, probability, or duration  
- Supports batch analysis of multiple species, regions, or time steps  
- Outputs results as shapefiles, GeoJSON, CSV, or graph objects  

---

## Directory Structure

```
CONNEX/
│
├── src/
  ├── connex/     # Core codebase
        │   ├── analysis/
                    │   ├── analysis.py    
                    │   ├── graph_builder.py
                    │   ├── metrics.py          
                    │   └── __init__.py         
        │   ├── examples/
                    │   ├── generate_example_data.py          
                    │   └── __init__.py  
        │   ├── plot/
                    │   ├── plot.py          
                    │   └── __init__.py    
        │   ├── preproc/
        │   ├── utils          
        │   └── __init__.py           
  ├── connex.egg_info/     
        │   ├── dependency_link.txt   
        │   ├── PKG-INFO
        │   ├── requires.txt
        │   ├── SOURCES.txt
        │   └── top_level.txt
├── notebooks/              # Jupyter tutorials and walkthroughs
├── output/                 # Generated graphs, metrics, maps
├── README.md           # This file
├── LICENSE
├── setup.py
├── environment.yml         # to set up a conda environmen 
└── requirements.txt        # Python dependencies
```

---

## Quickstart

1. **Install Connex Package** (Python 3.8+):

   Install the `connex` package and its dependencies using `pip`:

   ```bash
   pip install -e .
   ```
  This installs connex in editable mode for local development. If you're
  installing from PyPI, you can use:

  ```bash
  pip install connex
  ```
  we advise setting up a conda environment for your install. This can be done by installing conda and creating   an environment using:

  ```bash
 conda env create -n connex python=3.11
  ```
 or by using the provided environment.yml (which will also install all required dependencies:

 ```bash
 conda env create -f environment.yml
 ```
2. Prepare your data  
    Use larval dispersal outputs in **NetCDF**, .Zarr or **CSV** format. These should contain particle release and settlement coordinates, and optionally timestamps.

3. Generate node shapefile (optional)
  
  ```python
  from connex.shapefile_tools import generate_node_grid
  generate_node_grid(extent=[-75, -65, 10, 20], cell_size=0.5, output_path='data/nodes.shp')
  ```

4. Build connectivity graph

  ```python
  from connex.graph_builder import build_connectivity_graph
  G = build_connectivity_graph('data/larvae_dispersal.csv', 'data/nodes.shp')
  ```

5. Compute metrics and detect communities

  ```python
  from connex.metrics import compute_all_metrics
  results = compute_all_metrics(G, detect_communities=True)
  ```

6. Visualize the network

  ```python
  from connex.plot import plot_network
  plot_network(G, base_map='world')
  ```

---

## Example Metrics

| Metric           | Description                                 |
|------------------|---------------------------------------------|
| Degree centrality| Number of connections per node              |
| Betweenness      | Influence over dispersal pathways           |
| Clustering coef. | Local connectivity density                  |
| Modularity       | Community structure within the network      |
| Path length      | Connectivity efficiency                     |

---

## Use Cases

- Marine protected area (MPA) network design  
- Identifying critical habitat links for fishery species  
- Comparing dispersal under climate or current scenarios  
- Estimating metapopulation persistence based on connectivity  
- Detecting spatial communities of interconnected habitat  

---

## Requirements

- `networkx`  
- `pandas`  
- `matplotlib`  
- `geopandas`  
- `numpy`  
- `basemap` or `cartopy` (for mapping)

Can be installed seperately with:

```bash
pip install -r requirements.txt
```
 * Note that if you installed connex in a conda environment using the environment.yml these dependencies will     be installed
---

## Documentation & Tutorials

See the `notebooks/examples` folder for:

- Example input generation
- End-to-end examples  
- Node generation workflows  
- Multi-species comparisons  
- Custom map visualizations  

API documentation coming soon.

---

## Contributions

We welcome contributions. To suggest a feature, submit a pull request, or report a bug, please open an issue at:

https://github.com/mollyjames2/connex/issues

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use CONNEX in a publication, please cite:

> CONNEX: Connectivity Of Nodes and NEtwork eXploration.  
> James (2025). GitHub repository: https://github.com/mollyjames2/connex

