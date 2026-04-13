#!/usr/bin/env python3
"""
Interactive peak-calling test.  Point at a tracks.csv and pick a particle:

    python dev/test_peaks.py -f results/tracks.csv -p 0
    python dev/test_peaks.py -f results/tracks.csv -p 0 --signal mean_top_quartile --prominence 8 --smooth 1

If -p is omitted, the longest-tracked particle is used automatically.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file",       required=True,  help="Path to tracks.csv")
parser.add_argument("-p", "--particle",   type=int,       default=None,
                    help="Particle ID (default: longest track)")
parser.add_argument("--signal",           default="mean_intensity",
                    choices=["mean_intensity", "max_intensity", "mean_top_quartile"],
                    help="Which intensity column to use (default: mean_intensity)")
parser.add_argument("--fps",             type=float, default=20.0,
                    help="Frame rate in Hz (default: 20)")
parser.add_argument("--smooth",          type=int,   default=3,
                    help="Uniform-filter window size for AMPD (1 = no smoothing, default: 3)")
parser.add_argument("--prominence",      type=float, default=10,
                    help="scipy find_peaks prominence threshold (default: 10)")
args = parser.parse_args()

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(args.file)

if args.signal not in df.columns:
    available = [c for c in ["mean_intensity", "max_intensity", "mean_top_quartile"]
                 if c in df.columns]
    raise SystemExit(f"Column '{args.signal}' not in tracks.csv. Available: {available}")

if args.particle is None:
    args.particle = df.groupby("particle")["frame"].count().idxmax()
    print(f"No particle specified — using longest track: particle {args.particle}")

g = df[df["particle"] == args.particle].sort_values("frame")
if len(g) == 0:
    raise SystemExit(f"Particle {args.particle} not found in {args.file}")

frames_arr = g["frame"].values
raw        = g[args.signal].values.astype(float)


time_s   = frames_arr / args.fps
smoothed = uniform_filter1d(raw, size=args.smooth)

# ── scipy ──────────────────────────────────────────────────────────────────────
scipy_idx, scipy_props = find_peaks(raw, prominence=args.prominence)
dur = len(raw) / args.fps
print(f"scipy:  {len(scipy_idx)} peaks  |  rate = {len(scipy_idx) / dur:.3f} Hz  "
      f"(prominence={args.prominence}, no smoothing)")

# ── pyampd ─────────────────────────────────────────────────────────────────────
try:
    from pyampd.ampd import find_peaks as ampd_find_peaks
    ampd_raw = ampd_find_peaks(smoothed)
    n = len(smoothed)
    ampd_idx = np.array([
        idx for idx in ampd_raw
        if (idx == 0 or smoothed[idx] > smoothed[idx - 1])
        and (idx == n - 1 or smoothed[idx] > smoothed[idx + 1])
    ])
    print(f"ampd:   {len(ampd_idx)} peaks  |  rate = {len(ampd_idx) / dur:.3f} Hz  "
          f"(smooth window={args.smooth})")
    has_ampd = True
except ImportError:
    has_ampd = False
    print("pyampd not installed — skipping ampd detection")

print(f"\nParticle {args.particle}: {len(raw)} frames  {dur:.1f}s  "
      f"mean={raw.mean():.1f}  std={raw.std():.1f}  "
      f"min={raw.min():.1f}  max={raw.max():.1f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

for ax, sig, label in zip(axes, [raw, smoothed],
                          ["raw", f"smoothed (window={args.smooth})"]):
    ax.plot(time_s, sig, lw=1, color="steelblue", label=label)
    if len(scipy_idx):
        ax.plot(time_s[scipy_idx], sig[scipy_idx], "v", color="orange",
                markersize=7, label=f"scipy (n={len(scipy_idx)}, {len(scipy_idx)/dur:.2f} Hz)")
    if has_ampd and len(ampd_idx):
        ax.plot(time_s[ampd_idx], sig[ampd_idx], "^", color="red",
                markersize=5, label=f"ampd (n={len(ampd_idx)}, {len(ampd_idx)/dur:.2f} Hz)")
    ax.set_ylabel(args.signal, fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title(label)

axes[-1].set_xlabel("Time (s)")
plt.suptitle(f"Particle {args.particle}  |  {args.signal}  |  "
             f"prominence={args.prominence}  |  smooth={args.smooth}  |  fps={args.fps:.1f}")
plt.tight_layout()
plt.show()
