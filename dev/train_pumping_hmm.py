#!/usr/bin/env python3
"""
Train a 2-state Gaussian HMM for pumping state classification.

States: active (tight ~0.25 s IPIs) and quiescent (long IPIs > 1 s).
Input:  pumping_events.csv files from one or more result directories.
Output: models/pumping_hmm.pkl — replaces the existing model bundle.

Usage
-----
    python dev/train_pumping_hmm.py <results_dir> [--metadata meta.csv]
    python dev/train_pumping_hmm.py /path/to/reanalysis --metadata metadata.csv

The results directory is searched recursively for pumping_events.csv files.
A metadata CSV with columns folder_name, genotype, food, condition is optional
but recommended — it labels the per-condition diagnostic plots.

The quiescent IPI threshold is fixed at 1.0 s.  Any gap ≥ 1 s between
consecutive scipy-detected pumps is unambiguously a missed pumping event
at the frame rates used (20 fps → 1 s = 20 missed frames).
"""

import argparse
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hmmlearn import hmm

# ── Constants ────────────────────────────────────────────────────────────────
QUIESCENT_IPI_THRESH = 1.0   # seconds — unambiguous pump-miss boundary
MIN_SCIPY_EVENTS     = 8     # minimum scipy peaks to include a particle
STATE_NAMES          = ["quiescent", "active"]

# Biologically informed initialisation (log-IPI space)
#   quiescent: mean ~3 s  → log(3)  ≈  1.10
#   active:    mean ~0.25 s → log(0.25) ≈ -1.39
INIT_MEANS = np.array([[1.10], [-1.39]])
INIT_VARS  = np.array([[1.0],  [0.05]])   # quiescent is broad, active is tight


def load_events(results_dir, metadata_path=None):
    """Load all pumping_events.csv files under results_dir.

    Returns a DataFrame with columns:
        particle, time_s, method, folder_name
    and optionally: genotype, food, condition (if metadata supplied).
    """
    results_dir = Path(results_dir)
    event_files = sorted(results_dir.rglob("pumping_events.csv"))
    if not event_files:
        print(f"ERROR: no pumping_events.csv found under {results_dir}")
        sys.exit(1)

    frames = []
    for ef in event_files:
        folder_name  = ef.parent.parent.name   # condition folder, used for metadata join
        results_name = ef.parent.name          # unique per video (e.g. 01_test_1_results)
        df = pd.read_csv(ef)
        df["folder_name"]  = folder_name
        df["results_name"] = results_name
        frames.append(df)
    events = pd.concat(frames, ignore_index=True)

    # uid is unique per (video, particle): condition_folder/results_dir/particle
    events["uid"] = (events["folder_name"] + "__" +
                     events["results_name"] + "__" +
                     events["particle"].astype(str))

    n_particles = events["uid"].nunique()
    print(f"Loaded {len(event_files)} event files  "
          f"({len(events)} events, {n_particles} unique (video, particle) pairs)")

    if metadata_path and Path(metadata_path).is_file():
        meta = pd.read_csv(metadata_path)
        meta.columns = meta.columns.str.strip()
        meta = meta.apply(lambda c: c.str.strip() if c.dtype == object else c)
        events = events.merge(meta, on="folder_name", how="left")
        print(f"Metadata loaded — conditions: "
              f"{sorted(events['condition'].dropna().unique())}")

    return events


def build_ipi_sequences(events):
    """Extract per-particle scipy IPI sequences.

    Returns
    -------
    sequences : list of 1-D float arrays  (log-transformed IPIs)
    lengths   : list of int
    meta_rows : list of dicts  (uid, n_events, frac_above_thresh, condition)
    """
    scipy_ev = events[events["method"] == "scipy"].copy()
    sequences, lengths, meta_rows = [], [], []

    for uid, grp in scipy_ev.groupby("uid"):
        grp = grp.sort_values("time_s")
        times = grp["time_s"].values
        if len(times) < MIN_SCIPY_EVENTS:
            continue
        ipis = np.diff(times)
        log_ipis = np.log(ipis)
        sequences.append(log_ipis)
        lengths.append(len(log_ipis))
        row = {"uid": uid, "n_events": len(times),
               "frac_above_thresh": float((ipis > QUIESCENT_IPI_THRESH).mean())}
        if "condition" in grp.columns:
            row["condition"] = grp["condition"].iloc[0]
        if "genotype" in grp.columns:
            row["genotype"] = grp["genotype"].iloc[0]
        meta_rows.append(row)

    all_ipis = np.concatenate(sequences)
    print(f"\nTraining set: {len(sequences)} particles, "
          f"{len(all_ipis)} IPIs")
    print(f"  IPI range:    {np.exp(all_ipis.min()):.3f} – "
          f"{np.exp(all_ipis.max()):.3f} s")
    print(f"  IPI median:   {np.exp(np.median(all_ipis)):.3f} s")
    frac_q = float((np.exp(all_ipis) > QUIESCENT_IPI_THRESH).mean())
    print(f"  Frac > {QUIESCENT_IPI_THRESH} s: {frac_q*100:.1f}%")

    return sequences, lengths, pd.DataFrame(meta_rows)


def fit_hmm(sequences, lengths):
    """Fit a 2-state GaussianHMM and return the model."""
    X = np.concatenate(sequences).reshape(-1, 1)

    model = hmm.GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=200,
        tol=1e-5,
        random_state=42,
        init_params="st",   # initialise start probs and transitions from data
        params="stmc",      # learn everything
    )
    model.means_   = INIT_MEANS.copy()
    model.covars_  = INIT_VARS.copy().reshape(2, 1, 1)

    model.fit(X, lengths)
    print(f"\nHMM fitted  (converged: {model.monitor_.converged})")

    # Sort states so state 0 = quiescent (higher mean log-IPI)
    order = np.argsort(model.means_.flatten())[::-1]  # descending
    if list(order) != [0, 1]:
        print("  Re-ordering states: quiescent=0, active=1")
        model.means_  = model.means_[order]
        model.covars_ = model.covars_[order]
        model.transmat_ = model.transmat_[order][:, order]
        model.startprob_ = model.startprob_[order]

    print(f"\nFitted state parameters:")
    for i, name in enumerate(STATE_NAMES):
        mu  = float(model.means_[i])
        sig = float(np.sqrt(model.covars_[i].flatten()[0]))
        print(f"  {name:12s}  mean IPI = {np.exp(mu):.3f} s  "
              f"(log μ={mu:.3f}, σ={sig:.3f})")
    print(f"\nTransition matrix:")
    for i, name in enumerate(STATE_NAMES):
        row = "  ".join(f"{v:.4f}" for v in model.transmat_[i])
        print(f"  {name:12s} → [{row}]")

    return model


def make_diagnostic_plots(model, sequences, lengths, meta_df, output_path):
    """Save a multi-panel diagnostic figure."""
    all_ipis = np.concatenate(sequences)
    all_log  = all_ipis.reshape(-1, 1)

    # Decode state sequence
    X_concat = np.concatenate(sequences).reshape(-1, 1)
    states_concat = model.predict(X_concat, lengths)

    # Panel layout
    has_cond = "condition" in meta_df.columns
    n_rows = 2 + (1 if has_cond else 0)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    fig.suptitle("HMM training diagnostics — 2-state pumping model", fontsize=13)

    colors = {"quiescent": "#e07070", "active": "#5599cc"}

    # 1a. Log-IPI distribution with state assignments
    ax = axes[0, 0]
    for s_idx, s_name in enumerate(STATE_NAMES):
        mask = states_concat == s_idx
        ax.hist(all_ipis[mask], bins=80, alpha=0.55,
                color=colors[s_name], label=s_name, density=True)
    ax.axvline(np.log(QUIESCENT_IPI_THRESH), color="k", ls="--",
               label=f"threshold ({QUIESCENT_IPI_THRESH} s)")
    ax.set_xlabel("log(IPI) [s]")
    ax.set_ylabel("density")
    ax.set_title("Log-IPI by decoded state")
    ax.legend(fontsize=8)

    # 1b. Raw IPI distribution (linear scale, zoomed)
    ax = axes[0, 1]
    raw_ipis = np.exp(all_ipis)
    for s_idx, s_name in enumerate(STATE_NAMES):
        mask = states_concat == s_idx
        ax.hist(raw_ipis[mask], bins=100, alpha=0.55, range=(0, 3),
                color=colors[s_name], label=s_name, density=True)
    ax.axvline(QUIESCENT_IPI_THRESH, color="k", ls="--")
    ax.set_xlabel("IPI (s)")
    ax.set_ylabel("density")
    ax.set_title("IPI distribution (0–3 s)")
    ax.legend(fontsize=8)

    # 2a. Transition matrix heatmap
    ax = axes[1, 0]
    im = ax.imshow(model.transmat_, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(STATE_NAMES)
    ax.set_yticks([0, 1]); ax.set_yticklabels(STATE_NAMES)
    ax.set_xlabel("to state"); ax.set_ylabel("from state")
    ax.set_title("Transition matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{model.transmat_[i,j]:.3f}",
                    ha="center", va="center", fontsize=10,
                    color="white" if model.transmat_[i,j] > 0.5 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 2b. Per-particle frac_quiescent distribution
    ax = axes[1, 1]
    ax.hist(meta_df["frac_above_thresh"].values, bins=30, color="#5599cc",
            edgecolor="white", linewidth=0.3)
    ax.axvline(meta_df["frac_above_thresh"].median(), color="k", ls="--",
               label=f"median = {meta_df['frac_above_thresh'].median():.2f}")
    ax.set_xlabel(f"Fraction of IPIs > {QUIESCENT_IPI_THRESH} s")
    ax.set_ylabel("particles")
    ax.set_title("Per-particle quiescent fraction (training set)")
    ax.legend(fontsize=8)

    # 3. Per-condition frac_quiescent (if metadata available)
    if has_cond:
        ax = axes[2, 0]
        conds = sorted(meta_df["condition"].dropna().unique())
        for i, cond in enumerate(conds):
            vals = meta_df[meta_df["condition"] == cond]["frac_above_thresh"].values
            ax.scatter([i] * len(vals) + np.random.default_rng(42).uniform(-0.15, 0.15, len(vals)),
                       vals, alpha=0.5, s=20, color="#5599cc", linewidths=0)
            ax.plot([i - 0.3, i + 0.3], [np.median(vals)] * 2, color="k", lw=2)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(f"Frac IPIs > {QUIESCENT_IPI_THRESH} s")
        ax.set_title("Per-condition quiescent fraction")
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        axes[2, 1].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDiagnostic plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train 2-state pumping HMM from pumping_events.csv files"
    )
    parser.add_argument("results_dir",
                        help="Directory to search recursively for pumping_events.csv")
    parser.add_argument("--metadata", default=None,
                        help="CSV with columns folder_name, genotype, food, condition")
    parser.add_argument("--output", default=None,
                        help="Output .pkl path (default: models/pumping_hmm.pkl)")
    args = parser.parse_args()

    repo_root  = Path(__file__).parent.parent
    model_out  = Path(args.output) if args.output else repo_root / "models" / "pumping_hmm.pkl"
    plot_out   = model_out.with_name("pumping_hmm_diagnostics.png")

    # ── Load data ─────────────────────────────────────────────────────────────
    events  = load_events(args.results_dir, args.metadata)
    seqs, lengths, meta_df = build_ipi_sequences(events)

    if len(seqs) < 10:
        print(f"ERROR: only {len(seqs)} particles meet the minimum event count "
              f"({MIN_SCIPY_EVENTS}). Need more data.")
        sys.exit(1)

    # ── Fit ───────────────────────────────────────────────────────────────────
    model = fit_hmm(seqs, lengths)

    # ── Diagnostics ──────────────────────────────────────────────────────────
    make_diagnostic_plots(model, seqs, lengths, meta_df, plot_out)

    # ── Training statistics for bundle ───────────────────────────────────────
    all_ipis = np.concatenate([np.exp(s) for s in seqs])
    train_median = float(np.median(all_ipis))
    train_p10    = float(np.percentile(all_ipis, 10))
    train_p90    = float(np.percentile(all_ipis, 90))
    datasets     = sorted(events["folder_name"].unique().tolist())

    # ── Save bundle ──────────────────────────────────────────────────────────
    bundle = {
        "model":                 model,
        "state_names":           STATE_NAMES,
        "quiescent_ipi_thresh":  QUIESCENT_IPI_THRESH,
        "min_scipy_events":      MIN_SCIPY_EVENTS,
        "training_median_ipi":   train_median,
        "training_ipi_p10":      train_p10,
        "training_ipi_p90":      train_p90,
        "training_n_particles":  len(seqs),
        "training_n_intervals":  len(all_ipis),
        "training_datasets":     datasets,
        "training_date":         str(date.today()),
    }

    model_out.parent.mkdir(exist_ok=True)
    with open(model_out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nModel saved to {model_out}")
    print(f"  Particles:  {len(seqs)}")
    print(f"  IPIs:       {len(all_ipis)}")
    print(f"  Threshold:  {QUIESCENT_IPI_THRESH} s")
    print(f"  Trained:    {date.today()}")
    print(f"\nNow re-run track.py on your videos to generate new pumping_ipi_states.csv files.")


if __name__ == "__main__":
    main()
