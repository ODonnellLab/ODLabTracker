#!/usr/bin/env python3
"""
Diagnostic: visualize all detected objects in the first frame of a video.
Usage: python dev/inspect_first_frame.py -f <video> -c <config.yaml>
"""
import sys
import os
import argparse
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import imageio.v3 as iio
import tifffile
from skimage.color import rgb2gray

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ODLabTracker import tracking

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", required=True)
parser.add_argument("-c", "--config", required=True)
args = parser.parse_args()

# Load config
with open(args.config) as f:
    cfg = yaml.safe_load(f)

min_area      = cfg['min_area']
max_area      = cfg['max_area']
thresh        = None if cfg['thresh'] == 'None' else cfg['thresh']
illumination  = cfg['illumination']
thresh_method = cfg.get('thresh_method', 'otsu')
backsub       = cfg.get('backsub', False)
backsub_frames = cfg.get('backsub_frames', 10)
max_objects   = cfg.get('max_objects', None)

# Read all frames needed (lazily)
file_ext = os.path.splitext(args.filename)[1].lower()
if file_ext in {'.tif', '.tiff'}:
    tif_file = tifffile.TiffFile(args.filename)
    num_frames = len(tif_file.pages)
    def get_frame(i):
        f = tif_file.pages[i].asarray()
        return f.astype(np.uint8) if f.ndim == 2 else (rgb2gray(f) * 255).astype(np.uint8)
else:
    props = iio.improps(args.filename, plugin="pyav")
    num_frames = props.shape[0]
    def get_frame(i):
        f = iio.imread(args.filename, index=i, plugin="pyav")
        return (rgb2gray(f) * 255).astype(np.uint8) if f.ndim == 3 else f.astype(np.uint8)

frame = get_frame(0)

# Apply background subtraction if enabled (mirrors FastTrack.py logic)
if backsub:
    print(f"Applying background subtraction using {backsub_frames} frames...")
    frame_indices = np.linspace(1, num_frames - 1, backsub_frames, dtype=int)
    avg = np.zeros_like(frame, dtype=np.float32)
    for i in frame_indices:
        avg += get_frame(i).astype(np.float32)
    average_frame = (avg / backsub_frames).astype(np.uint8)
    frame = tracking.subtract_background(frame, average_frame)
    print(f"Background subtracted. Frame stats: min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")

# Run detection with NO area filter so we see everything
# (max_objects not applied here so we can see the full picture)
mask_all, props_all, global_thresh = tracking.preprocess_frame(
    frame, min_area=0, max_area=99999, thresh=thresh,
    illumination=illumination, thresh_method=thresh_method
)

# Show what threshold the binary search would choose (with area filter)
if max_objects is not None and thresh is None:
    _, _, tuned_thresh = tracking.preprocess_frame(
        frame, min_area=min_area, max_area=max_area, thresh=None,
        illumination=illumination, thresh_method=thresh_method,
        max_objects=max_objects
    )

# Separate into in-range and out-of-range
in_range  = [p for p in props_all if min_area <= p.area_convex <= max_area]
too_small = [p for p in props_all if p.area_convex < min_area]
too_large = [p for p in props_all if p.area_convex > max_area]

print(f"Threshold used:       {global_thresh}")
print(f"Total objects found:  {len(props_all)}")
print(f"  In range [{min_area}-{max_area}]: {len(in_range)}")
print(f"  Too small (<{min_area}):          {len(too_small)}")
print(f"  Too large (>{max_area}):         {len(too_large)}")
if max_objects is not None and thresh is None:
    print(f"\nmax_objects={max_objects}: binary-search threshold would be {tuned_thresh}")

areas_all = sorted([p.area_convex for p in props_all])
print(f"\nArea distribution:")
print(f"  Min:    {min(areas_all):.1f}")
print(f"  Median: {np.median(areas_all):.1f}")
print(f"  Mean:   {np.mean(areas_all):.1f}")
print(f"  Max:    {max(areas_all):.1f}")
print(f"\nLargest 30 areas: {areas_all[-30:]}")

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. Raw / backsub'd frame
axes[0].imshow(frame, cmap='gray')
axes[0].set_title("Background-subtracted frame" if backsub else "Raw frame")
axes[0].axis('off')

# 2. All detections coloured by category
# Cap too_small dots to 2000 to avoid rendering crash
MAX_DOTS = 2000
too_small_plot = too_small[:MAX_DOTS]
axes[1].imshow(frame, cmap='gray')
if too_small_plot:
    ys = [p.centroid[0] for p in too_small_plot]
    xs = [p.centroid[1] for p in too_small_plot]
    axes[1].plot(xs, ys, 'b.', markersize=2)
for p in in_range:
    minr, minc, maxr, maxc = p.bbox
    rect = mpatches.Rectangle((minc, minr), maxc-minc, maxr-minr,
                               linewidth=1, edgecolor='lime', facecolor='none')
    axes[1].add_patch(rect)
    axes[1].text(minc, minr-2, f"{p.area_convex:.0f}", color='lime', fontsize=5)
for p in too_large:
    minr, minc, maxr, maxc = p.bbox
    rect = mpatches.Rectangle((minc, minr), maxc-minc, maxr-minr,
                               linewidth=1, edgecolor='red', facecolor='none')
    axes[1].add_patch(rect)
    axes[1].text(minc, minr-2, f"{p.area_convex:.0f}", color='red', fontsize=5)
legend = [
    mpatches.Patch(color='lime', label=f'In range ({len(in_range)})'),
    mpatches.Patch(color='red',  label=f'Too large ({len(too_large)})'),
    mpatches.Patch(color='blue', label=f'Too small ({len(too_small)})'),
]
axes[1].legend(handles=legend, loc='upper right', fontsize=7)
axes[1].set_title(f"Detections (thresh={global_thresh})\nmin_area={min_area}, max_area={max_area}")
axes[1].axis('off')

# 3. Area histogram
areas_in    = [p.area_convex for p in in_range]
areas_small = [p.area_convex for p in too_small]
areas_large = [p.area_convex for p in too_large]
all_areas   = areas_small + areas_in + areas_large
if all_areas:
    bins = np.linspace(0, min(max(all_areas), max_area * 3), 40)
    axes[2].hist(areas_small, bins=bins, color='blue',  alpha=0.7, label=f'Too small ({len(too_small)})')
    axes[2].hist(areas_in,    bins=bins, color='lime',  alpha=0.7, label=f'In range ({len(in_range)})')
    axes[2].hist(areas_large, bins=bins, color='red',   alpha=0.7, label=f'Too large ({len(too_large)})')
    axes[2].axvline(min_area, color='lime', linestyle='--', linewidth=1.5, label=f'min_area={min_area}')
    axes[2].axvline(max_area, color='red',  linestyle='--', linewidth=1.5, label=f'max_area={max_area}')
axes[2].set_xlabel("Convex area (pixels)")
axes[2].set_ylabel("Count")
axes[2].set_title("Object area histogram")
axes[2].legend(fontsize=7)

plt.tight_layout()
# Save to repo root (directory containing this script's parent)
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path = os.path.join(repo_root, "first_frame_detection.png")
try:
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved figure to: {out_path}")
except Exception as e:
    print(f"\nFailed to save figure: {e}")
