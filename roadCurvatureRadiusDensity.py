#!/usr/bin/env python3
import math
import warnings
from typing import Iterable, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import substring
import folium
import branca.colormap as cm
import osmnx as ox

# ----------------------------
# Config — tweak if you want
# ----------------------------
CENTER_LAT = 46.94809   # Bern
CENTER_LON = 7.44744
RADIUS_M   = 1000      # 1 km
NETWORK_TYPE = "all"   # or 'all', 'drive_service', etc.
RESAMPLE_SPACING_M = 0.2  # densify geometry to ~0.2 m between vertices, use to compute the radius
MAX_RADIUS_M = 100      # cap for plotting; use None for no cap
MIN_RADIUS_M = 1.0       # discard pathological tiny radii from noisy data
BINS = 100                # histogram bins

# ----------------------------
# Helpers
# ----------------------------

def resample_linestring(ls: LineString, spacing: float) -> LineString:
    """
    Densify a LineString by interpolating points every `spacing` meters.
    Assumes the geometry is in a projected CRS (meters).
    """
    if ls.length <= spacing:
        return ls

    num_segments = max(2, int(math.floor(ls.length / spacing)) + 1)
    distances = np.linspace(0, ls.length, num_segments)
    pts = [ls.interpolate(d) for d in distances]
    # Ensure last vertex equals original end for numerical stability
    if pts[-1].coords[-1] != ls.coords[-1]:
        pts[-1] = Point(ls.coords[-1])
    return LineString(pts)


def triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Signed area of triangle (a,b,c). For curvature radius we use abs(area).
    """
    return 0.5 * ((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))

def radius_from_three_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Circumradius of triangle through points a,b,c in 2D.
    If points are nearly colinear, returns np.inf.
    """
    # side lengths
    ab = np.linalg.norm(b - a)
    bc = np.linalg.norm(c - b)
    ac = np.linalg.norm(c - a)

    # If any two points coincide or degenerate
    if ab < 1e-6 or bc < 1e-6 or ac < 1e-6:
        return np.inf

    area = abs(triangle_area(a, b, c))
    if area < 1e-9:
        return np.inf

    R = (ab * bc * ac) / (4.0 * area)
    return R

def radii_along_linestring(ls: LineString) -> List[float]:
    """
    Compute discrete curvature radii along a LineString using
    sliding triplets of consecutive vertices.
    Returns a list of radii (meters).
    """
    coords = np.asarray(ls.coords, dtype=float)
    if coords.shape[0] < 3:
        return []

    radii = []
    for i in range(1, coords.shape[0] - 1):
        a, b, c = coords[i-1], coords[i], coords[i+1]
        R = radius_from_three_points(a, b, c)
        radii.append(R)
    return radii

def extract_edge_linestring(u, v, data) -> LineString:
    """
    Get a LineString for an edge. OSMnx may store 'geometry' for curved
    edges. If absent, build a straight segment from node coordinates.
    """
    if "geometry" in data and isinstance(data["geometry"], LineString):
        return data["geometry"]
    # Fallback: build from (x, y) of endpoints
    return LineString([(data["x_u"], data["y_u"]), (data["x_v"], data["y_v"])])

# ----------------------------
# Main
# ----------------------------

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    ox.settings.use_graphml = False  # we don't persist; speed up a bit

    print("Downloading OSM road network…")
    G = ox.graph_from_point(
        (CENTER_LAT, CENTER_LON),
        dist=RADIUS_M,
        network_type=NETWORK_TYPE,
        simplify=True,
        retain_all=False
    )

    # Project to a metric CRS suitable for Switzerland (CH1903+ / LV95 EPSG:2056)
    print("Projecting graph to EPSG:2056 (meters)…")
    Gp = ox.project_graph(G, to_crs="EPSG:2056")

    # Get GeoDataFrames (edges include geometries)
    print("Converting graph to GeoDataFrames…")
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(Gp, nodes=True, edges=True, fill_edge_geometry=True)

    # Keep only edges with a LineString geometry
    print("Preparing edge geometries…")
    # Ensure we have coordinates of u/v in projected CRS for fallback lines
    node_xy = gdf_nodes[["x", "y"]].to_dict("index")
    edges = []
    for (u, v, k), row in gdf_edges.iterrows():
        data = row.to_dict()
        # add endpoint coordinates so fallback line can be built
        data["x_u"], data["y_u"] = node_xy[u]["x"], node_xy[u]["y"]
        data["x_v"], data["y_v"] = node_xy[v]["x"], node_xy[v]["y"]
        try:
            ls = extract_edge_linestring(u, v, data)
            if ls.length > 0:
                edges.append(ls)
        except Exception:
            continue


    print(f"Edges collected: {len(edges)}")

    # Resample and collect radii
    all_radii = []
    print("Resampling edges and computing radii…")
    for ls in edges:
        ls_rs = resample_linestring(ls, RESAMPLE_SPACING_M)
        radii = radii_along_linestring(ls_rs)
        if radii:
            all_radii.extend(radii)

    radii = np.array(all_radii, dtype=float)

    # Filter radii for plotting
    if MIN_RADIUS_M is not None:
        radii = radii[radii >= MIN_RADIUS_M]
    if MAX_RADIUS_M is not None:
        radii = radii[radii <= MAX_RADIUS_M]

    if radii.size == 0:
        print("No radii computed (empty set after filtering). Try increasing MAX_RADIUS_M or changing spacing.")
        return

    print(f"Computed {radii.size} curvature samples.")

    # Plot histogram (relative density)
    plt.figure(figsize=(9, 5))
    plt.hist(radii, bins=BINS, density=True)
    plt.xlabel("Radius of curvature (m)")
    plt.ylabel("Relative density")
    plt.title(f"Road curvature radii within {RADIUS_M/1000:.0f} km of Bern (spacing={RESAMPLE_SPACING_M} m)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"curvature_radii.png")

if __name__ == "__main__":
    main()
