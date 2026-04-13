#!/usr/bin/env python3
"""
Run track.py on all .avi/.tif files in a directory tree, N files at a time.

Each file runs in its own subprocess. Output is streamed to per-file log files
in a 'logs/' subfolder next to this script, so you can tail them independently.

Usage:
    python run_fasttrack_parallel.py <directory> -c <config.yaml>
    python run_fasttrack_parallel.py ./data -c configs/IR_medium.yaml -j 4

Notes:
    - Default workers: 4, or number of files if fewer.
    - Each tracking job loads the full video into RAM. Keep -j low enough
      that total memory stays comfortable (rule of thumb: 2–4 jobs per 16 GB RAM).
    - Progress of individual jobs can be monitored with:
          tail -f logs/<filename>.log
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

VIDEO_EXTENSIONS = {".avi", ".mp4", ".tif", ".tiff"}


def run_one(video_path, config, log_dir):
    """Run track.py on a single file, streaming output to a log file."""
    log_path = Path(log_dir) / (Path(video_path).stem + ".log")
    with open(log_path, "w") as log:
        result = subprocess.run(
            [sys.executable, "track.py", "-f", video_path, "-c", config],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True
        )
    return video_path, result.returncode, str(log_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run track.py in parallel on all video files in a directory tree"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to search for video files (default: current directory)"
    )
    parser.add_argument("-c", "--config", required=True,
                        help="Config YAML file to pass to track.py")
    parser.add_argument("-j", "--jobs", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    args = parser.parse_args()

    directory = Path(args.directory).resolve()
    if not directory.is_dir():
        print(f"ERROR: Directory not found: {directory}")
        sys.exit(1)

    config = str(Path(args.config).resolve())
    if not Path(config).is_file():
        print(f"ERROR: Config file not found: {config}")
        sys.exit(1)

    video_files = sorted(
        f for f in directory.rglob("*") if f.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_files:
        print(f"No video files found in {directory}")
        sys.exit(0)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    n_workers = min(args.jobs, len(video_files))
    print(f"Found {len(video_files)} file(s) — running {n_workers} at a time")
    print(f"Config:  {config}")
    print(f"Logs:    {log_dir.resolve()}/")
    print(f"Monitor: tail -f logs/<name>.log")
    print("-" * 60)

    n_ok, n_err = 0, 0
    futures = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for video in video_files:
            f = pool.submit(run_one, str(video), config, str(log_dir))
            futures[f] = video

        for future in as_completed(futures):
            video_path, returncode, log_path = future.result()
            name = Path(video_path).name
            if returncode == 0:
                n_ok += 1
                print(f"  OK      {name}")
            else:
                n_err += 1
                print(f"  FAILED  {name}  (see {log_path})")

    print("\n" + "=" * 60)
    print(f"Done: {n_ok} succeeded, {n_err} failed")
    if n_err:
        print(f"Check logs/ for details on failed jobs.")


if __name__ == "__main__":
    main()
