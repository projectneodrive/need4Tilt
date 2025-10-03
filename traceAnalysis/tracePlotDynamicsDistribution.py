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

# Admissible deltaT range for sampling
FREQ_NOMINAL = 10.0  # Hz
FREQ_MARGIN = 1.5    # Hz, tolerance
DT_MIN = 1.0 / (FREQ_NOMINAL + FREQ_MARGIN)
DT_MAX = 1.0 / (FREQ_NOMINAL - FREQ_MARGIN)

# Realistic limits (street car)
MAX_SPEED = 200  # km/h
MIN_SPEED = 0
MAX_VERTICAL_SPEED = 20  # m/s
MAX_LINEAR_ACC = 10  # m/s²


# -----------------------------
# Helper functions
# -----------------------------
def reject_outliers(df):
    """Reject rows that are impossible for a street car"""
    speed = np.sqrt(df['velocity_north']**2 + df['velocity_east']**2 + df['velocity_down']**2) * SPEED_CONVERSION
    mask = (speed >= MIN_SPEED) & (speed <= MAX_SPEED)
    mask &= (np.abs(df['velocity_down']) <= MAX_VERTICAL_SPEED)
    filtered_df = df[mask]
    n_outliers = len(df) - len(filtered_df)
    print(f"Removed {n_outliers} outliers ({100.0*n_outliers/len(df):.2f}% of dataset)")
    return filtered_df


def compute_linear_dynamics(df):
    """Compute speed, acceleration along NED directions, dropping invalid dt."""
    timestamps = df['timestamp'].values / 1e6  # seconds
    dt = np.diff(timestamps)
    valid_mask = (dt >= DT_MIN) & (dt <= DT_MAX)
    valid_idx = np.where(valid_mask)[0] + 1

    t_valid = timestamps[valid_idx]
    vel_valid = {col: df[col].values[valid_idx] for col in LINEAR_VEL_COLS}

    speed = np.sqrt(
        vel_valid['velocity_north'] ** 2 +
        vel_valid['velocity_east'] ** 2 +
        vel_valid['velocity_down'] ** 2
    )

    acc = {}
    for col in LINEAR_VEL_COLS:
        acc[col] = np.gradient(vel_valid[col], t_valid)

    return t_valid, speed, acc


def compute_angular_dynamics(df):
    """Compute angular velocity and acceleration, dropping invalid dt."""
    timestamps = df['timestamp'].values / 1e6
    dt = np.diff(timestamps)
    valid_mask = (dt >= DT_MIN) & (dt <= DT_MAX)
    valid_idx = np.where(valid_mask)[0] + 1

    t_valid = timestamps[valid_idx]
    ang_vel = {}
    ang_acc = {}
    for col in ANGLES_COLS:
        angle_valid = df[col].values[valid_idx]
        w = np.gradient(angle_valid, t_valid)
        alpha = np.gradient(w, t_valid)
        ang_vel[col] = w
        ang_acc[col] = alpha

    return t_valid, ang_vel, ang_acc


def compute_sampling_stats(timestamps):
    if len(timestamps) < 2:
        return np.nan, np.nan
    dt = np.diff(timestamps) / 1e6
    mean_freq = 1.0 / np.mean(dt)
    jitter = np.std(dt)
    return mean_freq, jitter


def print_stats(name, data):
    data = np.array(data)
    print(f"--- {name} ---")
    print(f"Mean:   {np.mean(data):.4f}")
    print(f"Median: {np.median(data):.4f}")
    print(f"Std:    {np.std(data):.4f}")
    print(f"Min:    {np.min(data):.4f}")
    print(f"Max:    {np.max(data):.4f}\n")


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

full_df = pd.concat(dfs, ignore_index=True)
print(f"Total number of samples: {len(full_df)}")

# -----------------------------
# Sampling stats summary
# -----------------------------
sampling_df = pd.DataFrame(sampling_stats)
sampling_df_clean = sampling_df.dropna(subset=['mean_freq', 'jitter'])
print("\nPer-file sampling statistics (valid files only):")
print(sampling_df_clean.describe())

# -----------------------------
# Reject outliers
# -----------------------------
full_df = reject_outliers(full_df)

# -----------------------------
# Compute dynamics
# -----------------------------
t_valid, speed, acc = compute_linear_dynamics(full_df)
speed_kmh = speed * SPEED_CONVERSION
vert_speed = full_df['velocity_down'].values
t_valid, ang_vel, ang_acc = compute_angular_dynamics(full_df)

# -----------------------------
# Print statistics
# -----------------------------
print_stats("Speed (km/h)", speed_kmh)
print_stats("Vertical speed (m/s)", vert_speed)
for col in LINEAR_VEL_COLS:
    print_stats(f"Linear acceleration {col} (m/s²)", acc[col])
for col in ANGLES_COLS:
    print_stats(f"Angular position {col} (rad)", full_df[col])
    print_stats(f"Angular velocity {col} (rad/s)", ang_vel[col])
    print_stats(f"Angular acceleration {col} (rad/s²)", ang_acc[col])

# -----------------------------
# Plot distributions
# -----------------------------
plt.figure(figsize=(15, 10))

def plot_hist_percent(ax, data, bins=50, color='b', alpha=0.5, label=None, xlim=None):
    counts, bin_edges = np.histogram(data, bins=bins)
    counts = counts / counts.sum() * 100
    ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge', color=color, alpha=alpha, label=label)
    if label:
        ax.legend()
    ax.set_ylabel('Frequency (%)')
    if xlim:
        ax.set_xlim(xlim)

# Speed
ax1 = plt.subplot(3, 3, 1)
plot_hist_percent(ax1, speed_kmh, bins=50, color='skyblue', xlim=(0, 130))
ax1.set_xlabel('Speed (km/h)')
ax1.set_title('Speed Distribution')

# Vertical speed
ax2 = plt.subplot(3, 3, 2)
plot_hist_percent(ax2, vert_speed, bins=50, color='salmon', xlim=(-5, 5))
ax2.set_xlabel('Vertical speed (m/s)')
ax2.set_title('Vertical Speed Distribution')

# Linear acceleration
ax3 = plt.subplot(3, 3, 3)
for col, color in zip(LINEAR_VEL_COLS, ['r','g','b']):
    plot_hist_percent(ax3, acc[col], bins=50, color=color, label=col, xlim=(-10, 10))
ax3.set_xlabel('Linear Acceleration (m/s²)')
ax3.set_title('Linear Acceleration Distribution')

# Angular positions (split)
ax = plt.subplot(3, 3, 4)
plot_hist_percent(ax, full_df["roll"], bins=1000, color='c', label="roll", xlim=(-0.2, 0.2))
ax.set_xlabel('roll (rad)')
ax.set_title('Angular Position: roll')

ax = plt.subplot(3, 3, 5)
plot_hist_percent(ax, full_df["pitch"], bins=1000, color='c', label="pitch", xlim=(-0.2, 0.2))
ax.set_xlabel('pitch (rad)')
ax.set_title('Angular Position: pitch')

ax = plt.subplot(3, 3, 6)
plot_hist_percent(ax, full_df["yaw"], bins=200, color='c', label="yaw", xlim=(-np.pi, np.pi))
ax.set_xlabel('yaw (rad)')
ax.set_title('Angular Position: yaw')

# Angular velocity
ax7 = plt.subplot(3, 3, 7)
for col, color in zip(ANGLES_COLS, ['r','g','b']):
    plot_hist_percent(ax7, ang_vel[col], bins=50, color=color, label=col, xlim=(-np.pi, np.pi))
ax7.set_xlabel('Angular velocity (rad/s)')
ax7.set_title('Angular Velocity Distribution')

# Angular acceleration
ax8 = plt.subplot(3, 3, 8)
for col, color in zip(ANGLES_COLS, ['r','g','b']):
    plot_hist_percent(ax8, ang_acc[col], bins=50, color=color, label=col, xlim=(-np.pi, np.pi))
ax8.set_xlabel('Angular acceleration (rad/s²)')
ax8.set_title('Angular Acceleration Distribution')

plt.tight_layout()
plt.show()
