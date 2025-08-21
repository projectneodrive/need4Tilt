#!/usr/bin/env python3
import warnings
import numpy as np
from shapely.geometry import LineString, Point
import folium
import branca.colormap as cm
import osmnx as ox

# ----------------------------
# Config
# ----------------------------
CENTER_LAT = 46.94809   # Bern
CENTER_LON = 7.44744
RADIUS_M = 1000         # 1 km radius
NETWORK_TYPE = "all"    # 'all', 'drive', 'walk', etc.
RESAMPLE_SPACING_M = 1  # 1 m spacing for curvature calculation
MIN_RADIUS_M = 1.0
MAX_RADIUS_M = 200

# ----------------------------
# Helpers
# ----------------------------

def resample_linestring(ls: LineString, spacing: float) -> LineString:
    """Densify a LineString by interpolating points every `spacing` meters."""
    if ls.length <= spacing:
        return ls
    num_points = max(2, int(np.floor(ls.length / spacing)) + 1)
    distances = np.linspace(0, ls.length, num_points)
    pts = [ls.interpolate(d) for d in distances]
    if pts[-1].coords[-1] != ls.coords[-1]:
        pts[-1] = Point(ls.coords[-1])
    return LineString(pts)

def triangle_area(a, b, c):
    """Signed area of triangle (a,b,c)."""
    return 0.5 * ((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))

def radius_from_three_points(a, b, c):
    """Circumradius of triangle through points a,b,c in 2D. Returns inf if nearly colinear."""
    ab = np.linalg.norm(b - a)
    bc = np.linalg.norm(c - b)
    ac = np.linalg.norm(c - a)
    if ab < 1e-6 or bc < 1e-6 or ac < 1e-6:
        return np.inf
    area = abs(triangle_area(a, b, c))
    if area < 1e-9:
        return np.inf
    return (ab * bc * ac) / (4.0 * area)

def radii_along_linestring(ls: LineString):
    coords = np.array(ls.coords, dtype=float)
    if coords.shape[0] < 3:
        return []
    radii = []
    for i in range(1, coords.shape[0]-1):
        a, b, c = coords[i-1], coords[i], coords[i+1]
        radii.append(radius_from_three_points(a, b, c))
    return radii

def extract_edge_linestring(u, v, data):
    if "geometry" in data and isinstance(data["geometry"], LineString):
        return data["geometry"]
    return LineString([(data["x_u"], data["y_u"]), (data["x_v"], data["y_v"])])

# ----------------------------
# Main
# ----------------------------

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    ox.settings.use_graphml = False

    print("Downloading OSM road network…")
    G = ox.graph_from_point(
        (CENTER_LAT, CENTER_LON),
        dist=RADIUS_M,
        network_type=NETWORK_TYPE,
        simplify=True,
        retain_all=False
    )

    print("Projecting graph to EPSG:2056…")
    Gp = ox.project_graph(G, to_crs="EPSG:2056")
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(Gp, nodes=True, edges=True, fill_edge_geometry=True)

    node_xy = gdf_nodes[["x","y"]].to_dict("index")
    edges = []
    for (u,v,k), row in gdf_edges.iterrows():
        data = row.to_dict()
        data["x_u"], data["y_u"] = node_xy[u]["x"], node_xy[u]["y"]
        data["x_v"], data["y_v"] = node_xy[v]["x"], node_xy[v]["y"]
        try:
            ls = extract_edge_linestring(u, v, data)
            if ls.length > 0:
                edges.append(ls)
        except:
            continue
    print(f"Collected {len(edges)} edges.")

    # Compute median radius per edge
    edge_radii = []
    for ls in edges:
        ls_rs = resample_linestring(ls, RESAMPLE_SPACING_M)
        rs = [r for r in radii_along_linestring(ls_rs) if np.isfinite(r)]
        if rs:
            median_r = np.median(rs)
            edge_radii.append((ls, median_r))

    if not edge_radii:
        print("No valid edges for curvature mapping.")
        return

    # Color mapping
    min_r = max(MIN_RADIUS_M, min(r for _, r in edge_radii))
    max_r = min(MAX_RADIUS_M, max(r for _, r in edge_radii))
    colormap = cm.linear.viridis.scale(min_r, max_r)
    colormap.caption = "Radius of curvature (m)"

    fmap = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=15, tiles="cartodbpositron")

    for ls, r in edge_radii:
        # convert projected coordinates back to lat/lon
        coords = [(y, x) for x, y in np.array(ls.coords)]
        color = colormap(min(max(r, min_r), max_r))
        folium.PolyLine(coords, color=color, weight=2, opacity=0.8).add_to(fmap)

    colormap.add_to(fmap)
    fmap.save("bern_curvature_map.html")
    print("Interactive map saved as bern_curvature_map.html.")

if __name__ == "__main__":
    main()
