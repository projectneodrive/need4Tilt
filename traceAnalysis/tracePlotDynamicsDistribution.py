#!/usr/bin/env python3
# tracePlotDynamicsDistribution.py

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
DATA_DIR = "../traceDataset/rtk"
SPEED_CONVERSION = 3.6  # m/s -> km/h

# Columns for analysis
LINEAR_VEL_COLS = ['velocity_north', 'velocity_east', 'velocity_down']
ANGLES_COLS = ['roll', 'pitch', 'yaw']


# -----------------------------
# Helper functions
# -----------------------------
def compute_sampling_stats(timestamps):
    """Compute mean frequency and jitter (per file)."""
    dt = np.diff(timestamps) / 1e6  # microseconds -> seconds
    mean_freq = 1.0 / np.mean(dt)
    jitter = np.std(dt)
    return mean_freq, jitter


def compute_linear_dynamics(df):
    """Compute speed, acceleration along NED directions."""
    # Linear speed magnitude (horizontal)
    speed = np.sqrt(df['velocity_north'] ** 2 + df['velocity_east'] ** 2 + df['velocity_down'] ** 2)
    # Linear accelerations (numerical derivative)
    dt = np.diff(df['timestamp']) / 1e6
    acc = {}
    for col in LINEAR_VEL_COLS:
        vel = df[col].values
        a = np.diff(vel) / dt
        acc[col] = a
    return speed, acc


def compute_angular_dynamics(df):
    """Compute angular velocity and angular acceleration."""
    dt = np.diff(df['timestamp']) / 1e6
    ang_vel = {}
    ang_acc = {}
    for col in ANGLES_COLS:
        angle = df[col].values
        w = np.diff(angle) / dt  # angular velocity
        ang_vel[col] = w
        alpha = np.diff(w) / dt[:-1]  # angular acceleration
        ang_acc[col] = alpha
    return ang_vel, ang_acc


def compute_sampling_stats(timestamps):
    """Compute mean frequency and jitter (per file). Handles short/empty traces."""
    if len(timestamps) < 2:
        return np.nan, np.nan  # Not enough samples to compute dt
    dt = np.diff(timestamps) / 1e6  # microseconds -> seconds
    mean_freq = 1.0 / np.mean(dt)
    jitter = np.std(dt)
    return mean_freq, jitter


# -----------------------------
# Load all CSV files
# -----------------------------
all_files = glob.glob(os.path.join(DATA_DIR, "**/rtk.csv"), recursive=True)
print(f"Found {len(all_files)} CSV files.")

dfs = []
sampling_stats = []

for file in all_files:
    df = pd.read_csv(file)
    dfs.append(df)

    mean_freq, jitter = compute_sampling_stats(df['timestamp'].values)
    sampling_stats.append({'file': file, 'mean_freq': mean_freq, 'jitter': jitter})

# Aggregate into a single DataFrame
full_df = pd.concat(dfs, ignore_index=True)
print(full_df)
print("------------------------------")
print(f"Total number of samples: {len(full_df)}")

# -----------------------------
# Sampling frequency summary
# -----------------------------
sampling_df = pd.DataFrame(sampling_stats)
# Remove files that had insufficient samples (NaN)
sampling_df_clean = sampling_df.dropna(subset=['mean_freq', 'jitter'])
print("\nPer-file sampling statistics (valid files only):")
print(sampling_df_clean.describe())

# -----------------------------
# Compute linear dynamics
# -----------------------------
speed, acc = compute_linear_dynamics(full_df)

# Horizontal speed distribution (km/h)
speed_kmh = speed * SPEED_CONVERSION

# Vertical speed (velocity_down)
vert_speed = full_df['velocity_down'].values

# -----------------------------
# Compute angular dynamics
# -----------------------------
ang_vel, ang_acc = compute_angular_dynamics(full_df)

# -----------------------------
# Plot distributions (normalized to 0-100%) with x-axis limits
# -----------------------------
plt.figure(figsize=(15, 10))

def plot_hist_percent(ax, data, bins=50, color='b', alpha=0.5, label=None, xlim=None):
    """Helper to plot histogram as % instead of counts with axis limits"""
    counts, bin_edges = np.histogram(data, bins=bins)
    counts = counts / counts.sum() * 100  # convert to %
    ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge', color=color, alpha=alpha, label=label)
    if label:
        ax.legend()
    ax.set_ylabel('Frequency (%)')
    if xlim:
        ax.set_xlim(xlim)

# Speed (0-130 km/h)
ax1 = plt.subplot(3, 3, 1)
plot_hist_percent(ax1, speed_kmh, bins=50, color='skyblue', xlim=(0, 130))
ax1.set_xlabel('Speed (km/h)')
ax1.set_title('Speed Distribution')

# Vertical speed (-5 to 5 m/s)
ax2 = plt.subplot(3, 3, 2)
plot_hist_percent(ax2, vert_speed, bins=50, color='salmon', xlim=(-5, 5))
ax2.set_xlabel('Vertical speed (m/s)')
ax2.set_title('Vertical Speed Distribution')

# Linear acceleration (-30 to 30 m/s²)
ax3 = plt.subplot(3, 3, 3)
for col, color in zip(LINEAR_VEL_COLS, ['r','g','b']):
    plot_hist_percent(ax3, acc[col], bins=50, color=color, label=col, xlim=(-30, 30))
ax3.set_xlabel('Linear Acceleration (m/s²)')
ax3.set_title('Linear Acceleration Distribution')

# Angular positions (split into 3 plots) limited to ±π

ax = plt.subplot(3, 3, 4)
plot_hist_percent(ax, full_df["roll"], bins=1000, color='c', label="roll", xlim=(-0.2, 0.2))
ax.set_xlabel('roll (rad)')
ax.set_title('Angular Position: roll')

ax = plt.subplot(3, 3, 5)
plot_hist_percent(ax, full_df["pitch"], bins=1000, color='c', label="pitch", xlim=(-0.2, 0.2))
ax.set_xlabel(f'pitch (rad)')
ax.set_title('Angular Position: pitch')

ax = plt.subplot(3, 3, 6)
plot_hist_percent(ax, full_df["yaw"], bins=200, color='c', label="yaw", xlim=(-np.pi, np.pi))
ax.set_xlabel('yaw (rad)')
ax.set_title('Angular Position: yaw')

# Angular velocity (-π to π rad/s)
ax7 = plt.subplot(3, 3, 7)
for col, color in zip(ANGLES_COLS, ['r','g','b']):
    plot_hist_percent(ax7, ang_vel[col], bins=50, color=color, label=col, xlim=(-np.pi, np.pi))
ax7.set_xlabel('Angular velocity (rad/s)')
ax7.set_title('Angular Velocity Distribution')

# Angular acceleration (-π to π rad/s²)
ax8 = plt.subplot(3, 3, 8)
for col, color in zip(ANGLES_COLS, ['r','g','b']):
    plot_hist_percent(ax8, ang_acc[col], bins=50, color=color, label=col, xlim=(-np.pi, np.pi))
ax8.set_xlabel('Angular acceleration (rad/s²)')
ax8.set_title('Angular Acceleration Distribution')

plt.tight_layout()
plt.show()