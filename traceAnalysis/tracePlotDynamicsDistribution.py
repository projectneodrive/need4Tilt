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

# Percentile cutoff for per-column filtering
PERCENTILE_LIMIT = 99.99  # adjustable


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


def filter_column_outliers(data, label, percentile=PERCENTILE_LIMIT):
    """Filter out values outside the percentile range, column-wise."""
    lower = np.nanpercentile(data, 100 - percentile)
    upper = np.nanpercentile(data, percentile)
    mask = (data >= lower) & (data <= upper)
    filtered = data[mask]
    n_total = len(data)
    n_kept = np.count_nonzero(mask)
    removed_pct = 100.0 * (n_total - n_kept) / n_total
    print(f"[{label}] Filtered {removed_pct:.3f}% ({n_total - n_kept}/{n_total}) outside [{lower:.3f}, {upper:.3f}] range.")
    return filtered


def plot_hist_percent(data, title, xlabel, bins=100, xlim=None):
    """Plot histogram in percentage with per-column outlier filtering."""
    data_filtered = filter_column_outliers(data, title)
    plt.figure(figsize=(8, 5))
    counts, bin_edges = np.histogram(data_filtered, bins=bins)
    counts = counts / counts.sum() * 100
    plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge', color='steelblue', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency (%)')
    if xlim:
        plt.xlim(xlim)
    plt.title(title)
    print_stats(title, data_filtered)
    plt.tight_layout()
    plt.show()


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
# Reject unrealistic rows
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
# Plot each distribution separately + log versions
# -----------------------------
print("\n=== DISTRIBUTIONS ===\n")

# Folder for saving plots
OUTPUT_DIR = "./plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Collect all subplot info
plot_data = []

def plot_and_save(data, title, xlabel, bins=100, xlim=None, logy=False, save=True):
    """Plot histogram (normal or log-y) and optionally save it."""
    data_filtered = filter_column_outliers(data, title)
    counts, bin_edges = np.histogram(data_filtered, bins=bins)
    counts = counts / counts.sum() * 100

    plt.figure(figsize=(8, 5))
    plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge', color='steelblue', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency (%)')
    if xlim:
        plt.xlim(xlim)
    if logy:
        plt.yscale('log')
        plt.ylabel('Frequency (%) [log]')
    plt.title(title + (" (log)" if logy else ""))

    print_stats(title + (" (log)" if logy else ""), data_filtered)

    plt.tight_layout()
    if save:
        safe_title = title.replace(" ", "_").replace("/", "_")
        fname = os.path.join(OUTPUT_DIR, f"{safe_title}{'_log' if logy else ''}.png")
        plt.savefig(fname, dpi=200)
        print(f"Saved: {fname}")
    plt.close()

    # Store for later combined plots
    plot_data.append({
        'data': data_filtered,
        'title': title,
        'xlabel': xlabel,
        'bins': bins,
        'xlim': xlim
    })


# ----------- Create and save individual plots -----------
# Speed and vertical speed
plot_and_save(speed_kmh, "Speed (km/h)", "Speed (km/h)", bins=100, xlim=(0, 130))
plot_and_save(speed_kmh, "Speed (km/h)", "Speed (km/h)", bins=100, xlim=(0, 130), logy=True)

plot_and_save(vert_speed, "Vertical Speed (m/s)", "Vertical Speed (m/s)", bins=100)
plot_and_save(vert_speed, "Vertical Speed (m/s)", "Vertical Speed (m/s)", bins=100, logy=True)

# Linear accelerations
for col in LINEAR_VEL_COLS:
    title = f"Linear Acceleration {col} (m/s²)"
    plot_and_save(acc[col], title, "Acceleration (m/s²)", bins=100)
    plot_and_save(acc[col], title, "Acceleration (m/s²)", bins=100, logy=True)

# Angular positions
for col in ANGLES_COLS:
    title = f"Angular Position {col} (rad)"
    plot_and_save(full_df[col], title, f"{col} (rad)", bins=500)
    plot_and_save(full_df[col], title, f"{col} (rad)", bins=500, logy=True)

# Angular velocities
for col in ANGLES_COLS:
    title = f"Angular Velocity {col} (rad/s)"
    plot_and_save(ang_vel[col], title, f"{col} (rad/s)", bins=100)
    plot_and_save(ang_vel[col], title, f"{col} (rad/s)", bins=100, logy=True)

# Angular accelerations
for col in ANGLES_COLS:
    title = f"Angular Acceleration {col} (rad/s²)"
    plot_and_save(ang_acc[col], title, f"{col} (rad/s²)", bins=1000)
    plot_and_save(ang_acc[col], title, f"{col} (rad/s²)", bins=1000, logy=True)


# -----------------------------
# Create condensed summary figures (normal + log)
# -----------------------------
def make_summary_plot(plot_data, logy=False, save_name="summary.png"):
    """Create a grid of subplots combining all histograms."""
    n = len(plot_data)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3))
    axes = axes.flatten()

    for i, pdict in enumerate(plot_data):
        ax = axes[i]
        counts, bin_edges = np.histogram(pdict['data'], bins=pdict['bins'])
        counts = counts / counts.sum() * 100
        ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge', color='steelblue', alpha=0.7)
        ax.set_title(pdict['title'], fontsize=9)
        ax.set_xlabel(pdict['xlabel'], fontsize=8)
        ax.set_ylabel('Frequency (%)', fontsize=8)
        if pdict['xlim']:
            ax.set_xlim(pdict['xlim'])
        if logy:
            ax.set_yscale('log')
            ax.set_ylabel('Frequency (%) [log]', fontsize=8)

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved summary plot: {save_path}")


# Save both normal and log summary figures
make_summary_plot(plot_data, logy=False, save_name="summary_normal.png")
make_summary_plot(plot_data, logy=True, save_name="summary_log.png")

