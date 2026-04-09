# ODLabTracker

A Python tool for tracking *C. elegans* behavior from video. Supports two analysis modes:

- **Postural** — tracks body posture and classifies movement states (forward runs, reversals, pirouettes)
- **Pumping** — tracks pharyngeal pumping rate from GCaMP or brightfield video using intensity peak detection

---

## Installation

### macOS / Linux

1. Clone the repository:

   ```bash
   git clone https://github.com/ODonnellLab/ODLabTracker.git
   cd ODLabTracker
   ```

2. Install in editable mode:

   ```bash
   pip install -e .
   ```

Requires Python 3.9+.

---

### Windows

The `av` (PyAV) package requires FFmpeg binaries that are not bundled in the standard pip wheel on Windows. The most reliable approach is to use [Miniconda](https://docs.anaconda.com/miniconda/) and install `av` via conda-forge before running pip.

**Step 1 — Install Miniconda**

Download and run the Miniconda installer for Windows from:
https://docs.anaconda.com/miniconda/

During installation, check "Add Miniconda3 to my PATH" (or use the **Anaconda Prompt** that is added to the Start menu).

**Step 2 — Clone the repository**

Install [Git for Windows](https://git-scm.com/download/win) if you don't have it, then in an Anaconda Prompt or PowerShell:

```
git clone https://github.com/ODonnellLab/ODLabTracker.git
cd ODLabTracker
```

**Step 3 — Create a conda environment**

```
conda create -n odlabtracker python=3.11
conda activate odlabtracker
```

**Step 4 — Install `av` via conda-forge**

```
conda install av -c conda-forge
```

**Step 5 — Install remaining dependencies**

```
pip install -e .
```

**Step 6 — Run the tracker**

Always activate the environment first in any new terminal session:

```
conda activate odlabtracker
python track.py -c configs\your_config.yaml -f path\to\video.avi
```

> **Note:** Use backslashes (`\`) for file paths on Windows, or wrap paths in quotes if they contain spaces.

---

## Usage

All analysis runs through the `track.py` entry point. The analysis mode is set by the `mode:` key in the config file.

```bash
python track.py -c configs/your_config.yaml -f path/to/video.avi
```

Omit `-f` to use an interactive file picker.

### Options

| Flag | Description |
|------|-------------|
| `-f` | Path to input video (`.avi`, `.mp4`, `.tif`/`.tiff`) |
| `-c` | Path to YAML config file (required) |
| `-v` | Verbose output |

---

## Config Files

Example configs are in `configs/`. Copy and edit the one closest to your setup.

| Config | Use case |
|--------|----------|
| `IR_medium.yaml` | IR brightfield, medium magnification |
| `PGlow_GCaMP_2.5x_8bin.yaml` | GCaMP pharynx, 2.5× objective, 8-bin camera |
| `PGlow_GCaMP_2.5x_16bin.yaml` | GCaMP pharynx, 2.5× objective, 16-bin camera |
| `Stereo0.5X_small.yaml` | Stereo scope, 0.5× objective |
| `Stereo1X_small.yaml` | Stereo scope, 1× objective |

### Common parameters (both modes)

```yaml
mode: postural        # postural or pumping
min_area: 100         # minimum object area in pixels
max_area: 5000        # maximum object area in pixels
gap_range: 5          # max frames a worm can disappear and still be re-linked
search_range: 30      # max pixel distance to link detections across frames
min_length: 10        # minimum track length to keep (frames)
frame_rate: 10        # frames per second
illumination: 1       # 0 = bright objects on dark bg (IR), 1 = dark on light
subsample: 1          # keep 1 in every N frames (use >1 for high-FPS video)
backsub: true         # subtract background before thresholding
backsub_frames: 10    # number of frames averaged for background model
pixel_length: 60      # pixels per mm (for speed calculations)
thresh: None          # manual threshold; None = auto (Otsu)
thresh_method: otsu   # auto-threshold method: otsu, triangle, yen, li
max_objects: null     # max detections per frame; threshold raised if exceeded
```

### Pumping-specific parameters

```yaml
mode: pumping
min_pump_track_frames: 20   # minimum track length for pumping analysis
peak_prominence: 10         # scipy find_peaks prominence threshold
stitch_gap_frames: 30       # max gap (frames) for stitching broken track fragments
stitch_gap_pixels: 40       # max distance (px) for stitching broken track fragments
```

---

## Outputs

Results are saved to a folder named `<video_name>_results/` next to the input video.

### Postural mode

| File | Description |
|------|-------------|
| `tracks.csv` | Per-frame centroid positions and motion statistics |
| `trajectories.png` | Overlay of all tracks on the first frame |

### Pumping mode

| File | Description |
|------|-------------|
| `tracks.csv` | Per-frame centroid positions and intensity measurements |
| `pumping_events.csv` | Frame index and method for each detected pump |
| `pumping_summary.csv` | Per-particle: track duration, pump count, mean rate (Hz) |
| `pixel_data.pkl` | Per-ROI pixel arrays keyed by `(particle, frame)` |
| `particle_<N>_pumping.mp4` | Annotated video for the best-tracked particle |

The pumping video shows:
- **Main view** — full video with a rate-coloured position trail (magenta >4 Hz, yellow 2–4 Hz, grey <2 Hz)
- **Right panel** — zoomed crop of the pharynx (globally normalised contrast) and a rolling intensity trace with red dots marking detected pumps

---

## Diagnostics

### Inspect first frame detection

```bash
python dev/inspect_first_frame.py -f path/to/video.avi -c configs/your_config.yaml
```

Shows the thresholded first frame with detected objects overlaid. Useful for tuning `min_area`, `max_area`, and `thresh`.

### Inspect pumping signal

```bash
python dev/inspect_pumping_signal.py -f path/to/video.avi -c configs/your_config.yaml --max-frames 200
```

Runs detection and linking on a short clip and plots the per-particle intensity trace with detected peaks. Useful for tuning `peak_prominence` before running the full analysis.

---

## Batch and parallel processing

```bash
# Process a list of files sequentially
python run_fasttrack_batch.py -c configs/your_config.yaml

# Process multiple files in parallel
python run_fasttrack_parallel.py -c configs/your_config.yaml
```

---

## Project structure

```
ODLabTracker/
├── track.py                  # Main entry point (dispatches by mode)
├── FastTrack.py              # Postural analysis
├── FastTrackPumping.py       # Pumping analysis
├── configs/                  # Example YAML config files
├── dev/                      # Diagnostic scripts
│   ├── inspect_first_frame.py
│   └── inspect_pumping_signal.py
├── src/ODLabTracker/
│   └── tracking.py           # Core tracking and analysis library
└── pyproject.toml
```
