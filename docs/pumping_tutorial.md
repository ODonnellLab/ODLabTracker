# Pumping Analysis Tutorial

This tutorial covers running pharyngeal pumping analysis end-to-end — from a raw video to per-particle state classification — using ODLabTracker's pumping mode.

---

## Prerequisites

Install ODLabTracker with all dependencies:

```bash
pip install -e .
```

This includes `hmmlearn`, which is required for HMM state classification. Verify:

```bash
python -c "import hmmlearn; print(hmmlearn.__version__)"
```

---

## Quick start

```bash
python track.py -c configs/PGlow_GCaMP_2.5x_8bin.yaml -f path/to/video.avi
```

Omit `-f` to use an interactive file picker.

---

## Config file setup

Pumping mode requires `mode: pumping` in your config. Copy the closest config from `configs/` and edit it for your setup.

```yaml
mode: pumping

# Detection
min_area: 30          # minimum pharynx area in pixels — lower for small/dim pharynges
max_area: 300         # maximum pharynx area in pixels
thresh: None          # None = auto-threshold (Otsu); set a fixed value if results are unstable
thresh_method: otsu   # otsu | triangle | yen | li
min_thresh: null      # floor for auto-threshold; null = auto-detect from background noise (2 × std of average frame); set explicitly to override
illumination: 0       # 0 = bright objects on dark background (GCaMP/IR); 1 = dark on light

# Linking
search_range: 15      # max pixel distance to link detections across frames
gap_range: 5          # max frames a pharynx can disappear and still be re-linked
min_length: 10        # minimum track length to keep (frames)

# Timing
frame_rate: 20        # frames per second — must match your acquisition rate
subsample: 1          # keep 1 in N frames; use 2–4 for very high-FPS recordings

# Background subtraction
backsub: true         # strongly recommended for GCaMP
backsub_frames: 20    # number of frames averaged to build background model

# Pumping detection
min_pump_track_frames: 20   # minimum track length for pumping analysis (frames)
peak_prominence: 10         # scipy find_peaks prominence — raise if too many false peaks
stitch_gap_frames: 30       # max gap (frames) for stitching broken track fragments
stitch_gap_pixels: 40       # max distance (px) for stitching broken track fragments
```

### Choosing a config

| Config | Use case |
|--------|----------|
| `PGlow_GCaMP_2.5x_8bin.yaml` | GCaMP pharynx marker, 2.5× objective, 8-bin camera |
| `PGlow_GCaMP_2.5x_16bin.yaml` | GCaMP pharynx marker, 2.5× objective, 16-bin camera |
| `PGlow_GCaMP_2.5x.yaml` | GCaMP pharynx marker, 2.5× objective, unbinned |
| `IR_medium.yaml` | IR brightfield, medium magnification |

---

## Running a batch

To process all videos in a directory:

```bash
# Sequential — one file at a time, live output
python run_fasttrack_batch.py /path/to/folder -c configs/PGlow_GCaMP_2.5x_8bin.yaml

# Parallel — N files simultaneously, output written to logs/
python run_fasttrack_parallel.py /path/to/folder -c configs/PGlow_GCaMP_2.5x_8bin.yaml -j 4
```

Both scripts search recursively for `.avi`, `.mp4`, `.tif`, and `.tiff` files.

For parallel processing, monitor a specific job with:

```bash
tail -f logs/video_name.log
```

---

## Outputs

Results are saved to `<video_name>_results/` in the same folder as the input video.

### File overview

| File | Description |
|------|-------------|
| `tracks.csv` | Per-frame centroid positions, area, and mean intensity for each tracked pharynx |
| `pumping_events.csv` | Each detected pump event: particle, frame, time, and detection method |
| `pumping_summary.csv` | Per-particle summary including peak counts, rates, and HMM state classification |
| `pixel_data.pkl` | Per-ROI pixel arrays keyed by `(particle, frame)` — for advanced analysis |
| `particle_<N>_pumping.mp4` | Annotated pumping video for the highest-quality tracked particle |

---

### `pumping_events.csv`

Each row is one detected pump peak.

| Column | Description |
|--------|-------------|
| `particle` | Particle (pharynx) ID |
| `frame` | Frame index of the peak |
| `time_s` | Time in seconds |
| `method` | Detection method: `scipy` or `ampd` |

Two methods are run in parallel:
- **scipy** — `scipy.signal.find_peaks` with a prominence threshold. Conservative; reliably detects the largest peaks and correctly produces long inter-peak gaps during genuine quiescence.
- **ampd** — Automatic Multiscale Peak Detection. More sensitive at high pumping rates (3–5 Hz) but may fire on noise during quiescent periods due to internal detrending.

---

### `pumping_summary.csv`

One row per tracked particle. Columns added by the HMM classifier are marked with ★.

#### Peak detection columns

| Column | Description |
|--------|-------------|
| `particle` | Particle ID |
| `track_duration_s` | Track length in seconds |
| `n_pumps_scipy` | Number of peaks detected by scipy |
| `mean_rate_hz_scipy` | Mean pumping rate (Hz) over full track — scipy |
| `n_pumps_ampd` | Number of peaks detected by AMPD |
| `mean_rate_hz_ampd` | Mean pumping rate (Hz) over full track — AMPD |
| `flag_censor` | `True` if either censoring flag is set |
| `flag_few_pumps` | `True` if either method detected fewer than 5 pumps |
| `flag_method_disagree` | `True` if scipy and AMPD counts differ by >20% |

#### ★ HMM state classification columns

| Column | Description |
|--------|-------------|
| `frac_quiescent` | Fraction of track time spent in quiescent state (scipy gaps > 0.42s) |
| `frac_slow` | Fraction of track time in slow-pumping state (HMM, ~3.2 Hz) |
| `frac_fast` | Fraction of track time in fast-pumping state (HMM, ~3.9 Hz) |
| `rate_slow_hz` | Mean pumping rate within slow-state intervals |
| `rate_fast_hz` | Mean pumping rate within fast-state intervals |
| `rate_active_ampd_hz` | AMPD-based rate during non-quiescent windows — most precise active rate |
| `flag_hmm_censored` | `True` if particle had fewer than 8 scipy events; all HMM columns `NaN` |

---

## Understanding the HMM state outputs

### The three states

The classifier uses a 3-state Gaussian HMM fitted to a reference dataset of 6 conditions (223 particles, ~12,000 inter-peak intervals). States are defined on the log-transformed inter-peak interval (IPI) sequence from scipy detections.

| State | Typical IPI | Typical rate | When observed |
|-------|-------------|--------------|---------------|
| Quiescent | >0.42s | <2.4 Hz | Worm not feeding; off food; behavioural pause |
| Slow | ~0.31s | ~3.2 Hz | Moderate feeding; transitional |
| Fast | ~0.26s | ~3.9 Hz | Active feeding; stimulated by food |

### The hybrid approach

- **Quiescence** is detected from **scipy** inter-peak gaps. Scipy's prominence threshold correctly produces long gaps during genuine pauses. AMPD should not be used for this because it fires on noise during quiescence due to internal detrending.
- **Active-period rate** (`rate_active_ampd_hz`) uses **AMPD** peaks within non-quiescent windows. AMPD detects more peaks at high rates where scipy can miss closely-spaced peaks.

### Interpreting the outputs

**`frac_quiescent`** is the primary metric for feeding state:
- Values near 0 → worm is actively pumping throughout
- Values near 1 → worm is mostly quiescent (off food, stressed, or mutant phenotype)

**`rate_active_ampd_hz`** is the best estimate of pumping speed when the worm is active. Compare this across conditions to detect rate differences independent of quiescence.

**`frac_slow` and `frac_fast`** capture within-track dynamics. Worms show sustained bouts in each state (median run length 10–22 intervals) rather than rapid flickering, so these fractions reflect genuine behavioural episodes.

**Example expected values:**

| Condition | `frac_quiescent` | `rate_active_ampd_hz` |
|-----------|-----------------|----------------------|
| WT on op50 (rich food) | ~0.35 | ~3.8 Hz |
| WT on jub39 | ~0.43 | ~3.5 Hz |
| WT off food 1h | ~0.71 | ~3.0 Hz |
| ser-7 off food 1h | ~0.89 | ~2.9 Hz |

### Censoring

Particles with fewer than 8 scipy-detected peaks are flagged with `flag_hmm_censored = True` and all HMM columns are set to `NaN`. These particles lack sufficient data for reliable HMM decoding. The `flag_few_pumps` column from the base analysis also identifies these particles.

### Boundary check warning

At the start of each run, the classifier checks whether the new data's IPI distribution is compatible with the training data:

```
[HMM] WARNING: batch median IPI (0.600s) is outside training range (0.250–0.550s).
Quiescent boundary (0.417s) may be misaligned.
```

This warning appears when:
- The batch median IPI falls outside the training range (0.25–0.55s), or
- Fewer than 1% or more than 80% of scipy IPIs exceed the quiescent threshold

If you see this warning, review the raw IPI distribution for your dataset before interpreting state fractions. The model may need retraining if your condition is far outside the original training range.

---

## Pumping video

The annotated video (`particle_<N>_pumping.mp4`) is generated for the highest-scoring particle (longest track × most detected pumps). It shows:

- **Left** — full video frame with a rate-coloured position trail:
  - Magenta: >4 Hz
  - Yellow: 2–4 Hz
  - Grey: <2 Hz
- **Right panel** — zoomed pharynx crop (globally normalised contrast across the full track) and a rolling intensity trace with red dots marking scipy-detected pump events

---

## Diagnostics

### Inspect detection on the first frame

```bash
python dev/inspect_first_frame.py -f path/to/video.avi -c configs/your_config.yaml
```

Shows the thresholded first frame with detected objects overlaid. Use to tune `min_area`, `max_area`, and `thresh`.

### Inspect the pumping signal

```bash
python dev/inspect_pumping_signal.py -f path/to/video.avi -c configs/your_config.yaml --max-frames 200
```

Runs detection and linking on a short clip and plots per-particle intensity traces with detected peaks. Use to tune `peak_prominence` before running the full analysis.

---

## Common issues

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Very few particles detected | `min_area` too high or `thresh` too low | Run `inspect_first_frame.py` to tune |
| Many false detections on off-food or sparse videos | Auto-threshold too low when `backsub=true` | Check `[diag] background noise σ` in output; set `min_thresh` explicitly if auto-computed value is wrong |
| Many false detections | `min_area` too low or noisy background | Raise `min_area`; enable `backsub` |
| `mean_rate_hz_scipy` near 0 but `mean_rate_hz_ampd` is high | scipy prominence too high | Lower `peak_prominence` |
| `flag_method_disagree` on most particles | Peak prominence mismatched to signal | Check signal with `inspect_pumping_signal.py` |
| All particles have `flag_hmm_censored = True` | Video too short or pumping rate too low | Increase video length; check signal quality |
| `[HMM] WARNING: batch median IPI outside training range` | Dataset far from training conditions | Interpret HMM outputs with caution; consider retraining |
| `rate_active_ampd_hz` is NaN for a classified particle | No AMPD peaks fell outside quiescent windows | Worm may be quiescent throughout; check `frac_quiescent` |
