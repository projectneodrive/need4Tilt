#!/usr/bin/env python3
# tracePlotMap.py

import os
import glob
import pandas as pd
import folium
from folium.plugins import Fullscreen, MiniMap
import numpy as np

# -----------------------------
# Parameters
# -----------------------------
DATA_DIR = "../traceDataset/rtk"
OUTPUT_HTML = "trace_map.html"

# -----------------------------
# Load all CSV files
# -----------------------------
all_files = glob.glob(os.path.join(DATA_DIR, "**/rtk.csv"), recursive=True)
print(f"Found {len(all_files)} CSV files.")

if not all_files:
    raise FileNotFoundError("No rtk.csv files found. Check DATA_DIR path.")

# -----------------------------
# Prepare map center
# -----------------------------
first_df = pd.read_csv(all_files[0])
center_lat = first_df["latitude"].iloc[0]
center_lon = first_df["longitude"].iloc[0]

m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap")

# Add utilities
Fullscreen().add_to(m)
MiniMap(toggle_display=True).add_to(m)

# -----------------------------
# Function to pick distinct colors
# -----------------------------
def get_color(i, n):
    cmap = plt.get_cmap("tab20", n)
    rgb = cmap(i)[:3]
    return f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"

# Fallback if matplotlib not available
try:
    import matplotlib.pyplot as plt
    colors = [get_color(i, len(all_files)) for i in range(len(all_files))]
except Exception:
    import random
    random.seed(42)
    colors = [f"#{random.randint(0,0xFFFFFF):06x}" for _ in all_files]

# -----------------------------
# Add each trace
# -----------------------------
for i, file in enumerate(all_files):
    try:
        df = pd.read_csv(file)
        if not {"latitude", "longitude"}.issubset(df.columns):
            print(f"Skipping {file} (missing lat/lon columns)")
            continue

        coords = df[["latitude", "longitude"]].dropna().values.tolist()

        if len(coords) < 2:
            print(f"Skipping {file} (not enough points)")
            continue

        # File name label
        short_name = os.path.basename(os.path.dirname(file))
        color = colors[i % len(colors)]

        folium.PolyLine(
            locations=coords,
            color=color,
            weight=3,
            opacity=0.8,
            popup=f"{short_name}",
        ).add_to(m)

        # Add start & end markers
        folium.Marker(coords[0], popup=f"Start: {short_name}", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(coords[-1], popup=f"End: {short_name}", icon=folium.Icon(color="red")).add_to(m)

        print(f"Added trace: {short_name} ({len(coords)} points)")

    except Exception as e:
        print(f"Error reading {file}: {e}")

# -----------------------------
# Save map
# -----------------------------
m.save(OUTPUT_HTML)
print(f"\n✅ Map saved as: {OUTPUT_HTML}\nOpen it in your browser to view the interactive traces.")
