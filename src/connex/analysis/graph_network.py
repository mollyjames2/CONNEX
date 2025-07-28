import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np
from shapely.geometry import Point
import os
from datetime import timedelta
import igraph as ig

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

    print(f" Path-based connectivity matrix built from {n_particles} particles.")
    return matrix
    


def connectivity_graph(
    matrix,
    community_algorithm="louvain",
    directed=True,
    remove_self_loops=True
):
    """
    Construct and analyze a connectivity graph from a matrix of node-to-node link strengths.

    Parameters:
    - matrix (pd.DataFrame): Square matrix with node IDs as index and columns (e.g., output of build_connectivity_matrix_*)
    - community_algorithm (str): One of ["louvain", "fastgreedy", "leiden"]
    - directed (bool): If True, treat the graph as directed; if False, undirected
    - remove_self_loops (bool): If True, zero out diagonal/self-recruitment links

    Returns:
    - metrics_df (pd.DataFrame): Table of graph metrics for each node
    - graph (igraph.Graph): The constructed graph object (for saving or plotting)
    """

    if remove_self_loops:
        matrix = matrix.copy()
        for node in matrix.index:
            matrix.at[node, node] = 0

    nodes = list(matrix.index)
    edge_list = []
    edge_weights = []

    for i, src in enumerate(nodes):
        for j, tgt in enumerate(nodes):
            weight = matrix.at[src, tgt]
            if weight > 0:
                edge_list.append((i, j))
                edge_weights.append(weight)

    # Build the graph
    graph = ig.Graph(directed=directed)
    graph.add_vertices(len(nodes))
    graph.vs["name"] = nodes
    graph.add_edges(edge_list)
    graph.es["weight"] = edge_weights

    # Compute centrality metrics
    pagerank = graph.pagerank(weights="weight", directed=directed)
    degree = graph.degree(mode="all")
    in_degree = graph.degree(mode="in")
    out_degree = graph.degree(mode="out")
    closeness = graph.closeness(weights="weight", mode="all", normalized=True)
    betweenness = graph.betweenness(weights="weight", directed=directed)
    eigenvector = graph.eigenvector_centrality(weights="weight", directed=directed)
    clustering = graph.transitivity_local_undirected(vertices=None, mode="zero")

    # Run community detection
    if community_algorithm == "fastgreedy":
        if directed:
            raise ValueError("FastGreedy only supports undirected graphs. Set directed=False.")
        communities = graph.community_fastgreedy(weights="weight").as_clustering()

    elif community_algorithm == "louvain":
        communities = graph.community_multilevel(weights="weight")

    elif community_algorithm == "leiden":
        try:
            import leidenalg
        except ImportError:
            raise ImportError("The 'leidenalg' package is required for Leiden community detection. Install it with `pip install leidenalg`.")
        communities = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition, weights="weight")

    else:
        raise ValueError(f"Unsupported community algorithm: {community_algorithm}")

    graph.vs["community"] = communities.membership

    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        "node": nodes,
        "pagerank": pagerank,
        "degree": degree,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "closeness": closeness,
        "betweenness": betweenness,
        "eigenvector": eigenvector,
        "clustering": clustering,
        "community": communities.membership
    })

    return metrics_df, graph



def summarize_connectivity_graph_metrics(metrics_df, output_file=None):
    """
    Generate a plain-English summary of graph network metrics from a larval dispersal connectivity matrix.

    Parameters:
    - metrics_df: DataFrame from the `connectivity_graph()` function, with one row per node.
    - output_file: Optional path to a .txt file to save the summary.

    Returns:
    - summary (str): A formatted text summary of the network structure.
    """
    lines = []
    total_nodes = len(metrics_df)
    isolated = metrics_df[metrics_df["degree"] == 0]
    connected = metrics_df[metrics_df["degree"] > 0]

    lines.append(f"\n The network contains {total_nodes} nodes representing release and/or settlement areas.")

    if not isolated.empty:
        iso_ids = ', '.join(str(n) for n in isolated['node'].tolist())
        lines.append(
            f"\n {len(isolated)} node(s) had no recorded larval exchange with any other nodes "
            f"\n (i.e., no larvae were observed dispersing from the node and settling in another node or settling into this node from another node). \n These were: {iso_ids}."
        )
    else:
        lines.append("\n All nodes participated in at least some level of larval dispersal.")

    # ---- Top PageRank ----
    max_pagerank = metrics_df["pagerank"].max()
    top_pagerank_nodes = metrics_df[metrics_df["pagerank"] == max_pagerank]["node"].tolist()
    node_list_str = ', '.join(str(n) for n in top_pagerank_nodes)
    lines.append(
        f"\n Node(s) {node_list_str} had the highest PageRank score ({max_pagerank:.3f}), "
        f"\n indicating strong overall influence in the network (likely acting as both larval sources and sinks)."
    )

    # ---- In-degree / Out-degree ----
    max_in = metrics_df["in_degree"].max()
    top_in_nodes = metrics_df[metrics_df["in_degree"] == max_in]["node"].tolist()
    max_out = metrics_df["out_degree"].max()
    top_out_nodes = metrics_df[metrics_df["out_degree"] == max_out]["node"].tolist()

    lines.append(
        f"\n Node(s) {', '.join(str(n) for n in top_in_nodes)} received larvae from the most nodes "
        f"(in-degree = {max_in}), \n while node(s) {', '.join(str(n) for n in top_out_nodes)} "
        f"exported larvae to the most others (out-degree = {max_out})."
    )

    # ---- Betweenness ----
    max_between = metrics_df["betweenness"].max()
    if max_between > 0:
        top_between_nodes = metrics_df[metrics_df["betweenness"] == max_between]["node"].tolist()
        lines.append(
            f"\n Node(s) {', '.join(str(n) for n in top_between_nodes)} had the highest betweenness centrality "
            f"\n ({max_between:.2f}), suggesting they acted as key stepping-stones or bridges "
            f"linking otherwise separated parts of the network."
        )
    else:
        lines.append("\n No nodes served as intermediaries (betweenness = 0), indicating limited dispersal corridors.")

    # ---- Community detection ----
    n_communities = metrics_df["community"].nunique()
    lines.append(
        f"\n The network contains {n_communities} community cluster(s), representing groups of nodes with high internal connectivity."
    )

    for cid in sorted(metrics_df["community"].unique()):
        members = metrics_df[metrics_df["community"] == cid]["node"].tolist()
        lines.append(f"  • Community {cid}: Nodes {', '.join(map(str, members))}")

    # ---- Output
    summary_text = "\n".join(lines)

    if output_file:
        with open(output_file, "w") as f:
            f.write(summary_text)
        print(f" Summary saved to: {output_file}")

    print(summary_text)
    return summary_text


