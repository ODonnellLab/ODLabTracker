#!/usr/bin/env python3
"""
Run track.py sequentially on all .avi/.tif files in a directory tree.

Usage:
    python run_fasttrack_batch.py <directory> -c <config.yaml>
    python run_fasttrack_batch.py ./data -c configs/IR_medium.yaml
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

VIDEO_EXTENSIONS = {".avi", ".mp4", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(
        description="Run track.py sequentially on all video files in a directory tree"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to search for video files (default: current directory)"
    )
    parser.add_argument("-c", "--config", required=True,
                        help="Config YAML file to pass to track.py")
    args = parser.parse_args()

    directory  = Path(os.path.abspath(args.directory))
    config     = os.path.abspath(args.config)
    video_files = sorted(
        f for f in directory.rglob("*") if f.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_files:
        print(f"No video files found in {directory}")
        sys.exit(0)

    print(f"Found {len(video_files)} file(s) — processing sequentially")
    print(f"Config: {config}")
    print("-" * 60)

    n_ok, n_err = 0, 0
    for i, video in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] {video}")
        result = subprocess.run(
            [sys.executable, "track.py", "-f", str(video), "-c", config],
            text=True
        )
        if result.returncode == 0:
            n_ok += 1
            print(f"  OK")
        else:
            n_err += 1
            print(f"  FAILED (exit code {result.returncode})")

    print("\n" + "=" * 60)
    print(f"Done: {n_ok} succeeded, {n_err} failed")


if __name__ == "__main__":
    main()
