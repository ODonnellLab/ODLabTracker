#!/usr/bin/env python3
"""
Diagnostic: plot mean intensity over time for tracked objects.
Runs detection + linking on a short clip, then plots per-particle
intensity signals so you can assess peak detection feasibility.

Usage:
  python dev/inspect_pumping_signal.py -f <video> -c <config.yaml>
  python dev/inspect_pumping_signal.py -f <video> -c <config.yaml> --max-frames 200
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v3 as iio
import tifffile
from skimage.color import rgb2gray
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ODLabTracker import tracking

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename",   required=True)
parser.add_argument("-c", "--config",     required=True)
parser.add_argument("--max-frames",       type=int, default=200,
                    help="Number of frames to analyse (default 200)")
parser.add_argument("--min-track-frac",   type=float, default=0.5,
                    help="Min fraction of max-frames a track must span to be plotted (default 0.5)")
args = parser.parse_args()

# ── Load config ────────────────────────────────────────────────────────────────
with open(args.config) as f:
    cfg = yaml.safe_load(f)

min_area       = cfg['min_area']
max_area       = cfg['max_area']
gap_range      = cfg['gap_range']
thresh         = None if cfg['thresh'] == 'None' else cfg['thresh']
thresh_method  = cfg.get('thresh_method', 'otsu')
search_range   = cfg['search_range']
min_length     = cfg['min_length']
frame_rate     = cfg['frame_rate']
illumination   = cfg['illumination']
backsub        = cfg.get('backsub', False)
backsub_frames = cfg.get('backsub_frames', 10)
max_objects    = cfg.get('max_objects', None)
peak_prominence = cfg.get('peak_prominence', 10)
max_frames     = args.max_frames

# ── Read frames ────────────────────────────────────────────────────────────────
file_ext   = os.path.splitext(args.filename)[1].lower()
iio_plugin = "tifffile" if file_ext in {'.tif', '.tiff'} else "pyav"

print(f"Reading up to {max_frames} frames using {iio_plugin}...")

if iio_plugin == "tifffile":
    tif_file   = tifffile.TiffFile(args.filename)
    total      = len(tif_file.pages)
    n          = min(max_frames, total)
    frames_raw = [tif_file.pages[i].asarray() for i in range(n)]
else:
    frames_raw = []
    for i, frame in enumerate(iio.imiter(args.filename, plugin=iio_plugin)):
        if i >= max_frames:
            break
        frames_raw.append(frame)

frames = []
for frame in frames_raw:
    if frame.ndim == 3 and frame.shape[-1] == 3:
        frame = (rgb2gray(frame) * 255).astype(np.uint8)
    else:
        frame = frame.astype(np.uint8)
    frames.append(frame)

print(f"Loaded {len(frames)} frames, shape {frames[0].shape}")

# ── Background subtraction ─────────────────────────────────────────────────────
if backsub:
    print(f"Applying background subtraction ({backsub_frames} frames)...")
    frame_indices = np.linspace(1, len(frames) - 1, backsub_frames, dtype=int)
    avg = np.zeros_like(frames[0], dtype=np.float32)
    for i in frame_indices:
        avg += frames[i].astype(np.float32)
    average_frame = (avg / backsub_frames).astype(np.uint8)
    frames = [tracking.subtract_background(f, average_frame) for f in frames]
    print(f"Background subtracted. Frame[0] stats: min={frames[0].min()}, max={frames[0].max()}, mean={frames[0].mean():.1f}")

# ── Threshold ──────────────────────────────────────────────────────────────────
if thresh is None:
    _, _, global_thresh = tracking.preprocess_frame(
        frames[0], min_area, max_area, None,
        illumination=illumination, thresh_method=thresh_method,
        max_objects=max_objects)
    print(f"Auto threshold: {global_thresh}")
else:
    global_thresh = thresh
    print(f"Manual threshold: {global_thresh}")

# ── Detect + link ──────────────────────────────────────────────────────────────
print("Running detection and linking...")
detections, _ = tracking.collect_detections_pumping(
    frames, global_thresh=global_thresh,
    min_area=min_area, max_area=max_area, illumination=illumination)

tracks = tracking.link_tracks(detections, search_range=search_range,
                               memory=gap_range, quiet=True)
tracks = tracking.filter_short_tracks(tracks, min_length=min_length)
tracks.drop(columns=["_det_id"], inplace=True, errors="ignore")

print(f"Particles after filtering: {tracks['particle'].nunique()}")

# Stitch broken track fragments
stitch_gap_frames = cfg.get('stitch_gap_frames', gap_range)
stitch_gap_pixels = cfg.get('stitch_gap_pixels', search_range * 2)
print(f"Stitching tracks (max_gap={stitch_gap_frames} frames, max_dist={stitch_gap_pixels} px)")
tracks, n_stitched = tracking.stitch_tracks(tracks,
                                            max_gap_frames=stitch_gap_frames,
                                            max_gap_pixels=stitch_gap_pixels)
tracks = tracking.filter_short_tracks(tracks, min_length=min_length)
tracks.drop(columns=["_det_id"], inplace=True, errors="ignore")
print(f"Particles after stitching: {tracks['particle'].nunique()}")

# ── Select particles to plot ───────────────────────────────────────────────────
min_len_plot = int(args.min_track_frac * max_frames)
counts = tracks.groupby("particle")["frame"].count()
plot_particles = counts[counts >= min_len_plot].index.tolist()

if not plot_particles:
    # Fall back: just take the 5 longest tracks
    plot_particles = counts.nlargest(5).index.tolist()
    print(f"No tracks span ≥{min_len_plot} frames — plotting top {len(plot_particles)} longest tracks")
else:
    print(f"Plotting {len(plot_particles)} particles spanning ≥{min_len_plot} frames")

# ── Plot ───────────────────────────────────────────────────────────────────────
n_particles = len(plot_particles)

# Assign a consistent colour per particle for both panels
cmap   = plt.cm.tab10
colors = {pid: cmap(i % 10) for i, pid in enumerate(plot_particles)}

try:
    from pyampd.ampd import find_peaks as ampd_find_peaks
    has_ampd = True
except Exception:
    has_ampd = False

# Figure: left column = intensity traces, right column = x/y map
fig = plt.figure(figsize=(18, 3.5 * max(n_particles, 1)))
gs  = fig.add_gridspec(n_particles, 2, width_ratios=[3, 1], hspace=0.45, wspace=0.3)

# Shared x-axis across intensity panels
intensity_axes = [fig.add_subplot(gs[i, 0]) for i in range(n_particles)]
for i in range(1, n_particles):
    intensity_axes[i].sharex(intensity_axes[0])

# Single shared x/y map on the right (all particles overlaid)
ax_xy = fig.add_subplot(gs[:, 1])

for i, pid in enumerate(plot_particles):
    ax   = intensity_axes[i]
    col  = colors[pid]
    g    = tracks[tracks["particle"] == pid].sort_values("frame")
    frames_arr = g["frame"].values
    intensity  = g["mean_intensity"].values
    time_s     = frames_arr / frame_rate
    x_pos      = g["x"].values
    y_pos      = g["y"].values

    # Detection gaps
    full_range   = np.arange(frames_arr[0], frames_arr[-1] + 1)
    missing      = np.setdiff1d(full_range, frames_arr)
    missing_time = missing / frame_rate

    # Peak detection
    scipy_peaks, _ = find_peaks(intensity, prominence=peak_prominence)

    # ── Intensity panel ──────────────────────────────────────────────────────
    ax.plot(time_s, intensity, color=col, lw=1, label="mean intensity")

    if len(missing) > 0:
        for mt in missing_time:
            ax.axvline(mt, color="red", alpha=0.35, lw=0.7)
        ax.axvline(missing_time[0], color="red", alpha=0.35, lw=0.7,
                   label=f"dropouts (n={len(missing)})")

    if len(scipy_peaks) > 0:
        ax.plot(time_s[scipy_peaks], intensity[scipy_peaks],
                "v", color="orange", markersize=7,
                label=f"scipy peaks (n={len(scipy_peaks)})")

    if has_ampd:
        ampd_peaks = ampd_find_peaks(intensity)
        if len(ampd_peaks) > 0:
            ax.plot(time_s[ampd_peaks], intensity[ampd_peaks],
                    "^", color="red", markersize=5,
                    label=f"ampd peaks (n={len(ampd_peaks)})")

    ax.set_ylabel("Mean intensity", fontsize=8)
    ax.set_title(f"Particle {pid}  |  {len(g)} frames  |  "
                 f"{len(missing)} gaps ({100*len(missing)/len(full_range):.1f}%)",
                 fontsize=8)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── X/Y trajectory panel ─────────────────────────────────────────────────
    # Colour trajectory by time (start=light, end=dark)
    points = np.array([x_pos, y_pos]).T.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    from matplotlib.collections import LineCollection
    lc = LineCollection(segs, cmap="Blues",
                        norm=plt.Normalize(time_s[0], time_s[-1]),
                        linewidth=1.5, alpha=0.8)
    lc.set_array(time_s[:-1])
    ax_xy.add_collection(lc)
    ax_xy.plot(x_pos[0],  y_pos[0],  "o", color=col, markersize=6,
               label=f"P{pid} start")
    ax_xy.plot(x_pos[-1], y_pos[-1], "s", color=col, markersize=6)
    ax_xy.text(x_pos[0], y_pos[0] - 8, str(pid), color=col,
               fontsize=7, ha="center")

intensity_axes[-1].set_xlabel("Time (s)", fontsize=8)

ax_xy.set_xlabel("X (pixels)", fontsize=8)
ax_xy.set_ylabel("Y (pixels)", fontsize=8)
ax_xy.set_title("X/Y trajectories\n(blue=early → dark=late)", fontsize=8)
ax_xy.invert_yaxis()   # match image coordinates
ax_xy.legend(fontsize=6, loc="upper right")
ax_xy.grid(True, alpha=0.3)
ax_xy.set_aspect("equal")

plt.suptitle(f"Pumping signal diagnostic — {os.path.basename(args.filename)}\n"
             f"thresh={global_thresh}  backsub={backsub}  "
             f"gap_range={gap_range}  search_range={search_range}",
             fontsize=9)

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path  = os.path.join(repo_root, "pumping_signal_diagnostic.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved to: {out_path}")

# ── Print stats ────────────────────────────────────────────────────────────────
print("\nPer-particle intensity stats:")
for pid in plot_particles:
    g          = tracks[tracks["particle"] == pid].sort_values("frame")
    frames_arr = g["frame"].values
    full_range = np.arange(frames_arr[0], frames_arr[-1] + 1)
    missing    = np.setdiff1d(full_range, frames_arr)
    peaks, _   = find_peaks(g["mean_intensity"].values, prominence=peak_prominence)
    dur        = len(frames_arr) / frame_rate
    rate       = len(peaks) / dur if dur > 0 else 0
    pct_missing = 100 * len(missing) / len(full_range)
    print(f"  Particle {pid:4d}: mean={g['mean_intensity'].mean():.1f}  "
          f"max={g['mean_intensity'].max():.1f}  "
          f"std={g['mean_intensity'].std():.1f}  "
          f"gaps={len(missing)} ({pct_missing:.1f}%)  "
          f"peaks(scipy)={len(peaks)}  rate={rate:.2f} Hz")
