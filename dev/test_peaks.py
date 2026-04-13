#!/usr/bin/env python3
"""
Interactive peak-calling test.  Paste intensity values from a single
particle (from tracks.csv) into the DATA block below, then run:

    python dev/test_peaks.py

Adjust PARAMS to tune smoothing, prominence, and which signal to use.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

# ── PASTE DATA HERE ────────────────────────────────────────────────────────────
# Copy a column from tracks.csv for one particle, e.g.:
#   tracks[(tracks.particle == 1)]["mean_intensity"].tolist()

MEAN_INTENSITY = [
    # paste values here, one per line
]

MAX_INTENSITY = [
    # paste values here (optional — leave empty to skip)
]

MEAN_TOP_QUARTILE = [
    # paste values here (optional — leave empty to skip)
]

FRAME_RATE = 20  # fps — used to convert frame index to time axis

# ── PARAMS ─────────────────────────────────────────────────────────────────────
SIGNAL         = "mean_intensity"   # "mean_intensity" | "max_intensity" | "mean_top_quartile"
SMOOTH_WINDOW  = 3                  # uniform filter size (frames); 1 = no smoothing
PROMINENCE     = 10                 # scipy find_peaks prominence threshold
# ──────────────────────────────────────────────────────────────────────────────

signals = {
    "mean_intensity":    MEAN_INTENSITY,
    "max_intensity":     MAX_INTENSITY,
    "mean_top_quartile": MEAN_TOP_QUARTILE,
}

raw = np.array(signals[SIGNAL], dtype=float)
if raw.size == 0:
    raise ValueError(f"No data in {SIGNAL} — paste values into the DATA block above.")

time_s   = np.arange(len(raw)) / FRAME_RATE
smoothed = uniform_filter1d(raw, size=SMOOTH_WINDOW)

# ── scipy ──────────────────────────────────────────────────────────────────────
scipy_idx, scipy_props = find_peaks(smoothed, prominence=PROMINENCE)
print(f"scipy:  {len(scipy_idx)} peaks  |  "
      f"rate = {len(scipy_idx) / (len(raw) / FRAME_RATE):.3f} Hz")

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
    print(f"ampd:   {len(ampd_idx)} peaks  |  "
          f"rate = {len(ampd_idx) / (len(raw) / FRAME_RATE):.3f} Hz")
    has_ampd = True
except ImportError:
    has_ampd = False
    print("pyampd not installed — skipping ampd detection")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

for ax, sig, label in zip(axes, [raw, smoothed],
                          ["raw", f"smoothed (window={SMOOTH_WINDOW})"]):
    ax.plot(time_s, sig, lw=1, color="steelblue", label=label)
    if len(scipy_idx):
        ax.plot(time_s[scipy_idx], sig[scipy_idx], "v", color="orange",
                markersize=7, label=f"scipy (n={len(scipy_idx)})")
    if has_ampd and len(ampd_idx):
        ax.plot(time_s[ampd_idx], sig[ampd_idx], "^", color="red",
                markersize=5, label=f"ampd (n={len(ampd_idx)})")
    ax.set_ylabel(SIGNAL, fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title(label)

axes[-1].set_xlabel("Time (s)")
plt.suptitle(f"{SIGNAL}  |  prominence={PROMINENCE}  |  smooth={SMOOTH_WINDOW}  |  fps={FRAME_RATE}")
plt.tight_layout()
plt.show()
