#!/usr/bin/env python3
"""
Convert all image series in a LIF file to AVI using JPEG compression.

Usage:
    python lif_to_avi.py path/to/file.lif
    python lif_to_avi.py path/to/file.lif --fps 20 --quality 90
    python lif_to_avi.py                   # interactive file picker
"""
import argparse
import os
import sys
import numpy as np
import cv2
from readlif.reader import LifFile
from tqdm import tqdm


def lif_to_avi(lif_path, fps=20, quality=85):
    output_dir = os.path.splitext(lif_path)[0] + "_avi"
    os.makedirs(output_dir, exist_ok=True)

    lif = LifFile(lif_path)
    series_list = lif.image_list
    print(f"Found {len(series_list)} series in {os.path.basename(lif_path)}")

    for idx, series_meta in enumerate(series_list):
        name     = series_meta.get("name", f"series_{idx + 1}")
        n_frames = series_meta["dims"].t
        print(f"\n[{idx + 1}/{len(series_list)}] {name}  ({n_frames} frames)")

        lif_image  = lif.get_image(idx)
        first      = np.array(lif_image.get_frame(z=0, t=0))

        if first.ndim == 3:
            first = first.mean(axis=2)

        h, w    = first.shape
        out_path = os.path.join(output_dir, f"{idx + 1:02d}_{name}.avi")

        # Determine global scale from LIF metadata bit depth so all frames
        # share the same contrast mapping (per-frame normalize caused flickering).
        # lif_image.bit_depth is a tuple of per-channel bit depths, e.g. (12,).
        # This correctly distinguishes 12-bit-in-uint16 from true 16-bit.
        bits = lif_image.bit_depth[0]
        global_min, global_max = 0, (2 ** bits) - 1
        print(f"  Bit depth: {bits}-bit  (scale range 0–{global_max})")

        def to_uint8(arr):
            arr = arr.astype(np.float32)
            arr = (arr - global_min) / (global_max - global_min) * 255.0
            return np.clip(arr, 0, 255).astype(np.uint8)

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=False)
        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, quality)

        for t in tqdm(range(n_frames), desc=name, unit="frame"):
            try:
                frame = np.array(lif_image.get_frame(z=0, t=t))
            except Exception as e:
                print(f"  Warning: skipping frame {t}: {e}")
                continue

            if frame.ndim == 3:
                frame = frame.mean(axis=2)
            if frame.dtype != np.uint8:
                frame = to_uint8(frame)

            writer.write(frame)

        writer.release()
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  Saved {out_path}  ({size_mb:.1f} MB)")

    print(f"\nDone. AVIs written to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LIF series to MJPEG AVI files")
    parser.add_argument("lif_file", nargs="?", help="Path to .lif file")
    parser.add_argument("--fps",     type=float, default=20,
                        help="Output frame rate (default: 20)")
    parser.add_argument("--quality", type=int,   default=85,
                        help="JPEG quality 0-100 (default: 85)")
    args = parser.parse_args()

    if args.lif_file:
        lif_path = os.path.abspath(args.lif_file)
    else:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        lif_path = filedialog.askopenfilename(
            title="Select a LIF file",
            filetypes=[("Leica LIF", "*.lif"), ("All files", "*.*")]
        )
        root.destroy()
        if not lif_path:
            print("No file selected.")
            sys.exit(1)

    if not os.path.isfile(lif_path):
        print(f"File not found: {lif_path}")
        sys.exit(1)

    lif_to_avi(lif_path, fps=args.fps, quality=args.quality)
