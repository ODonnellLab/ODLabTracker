import numpy as np
import imageio.v2 as imageio
from readlif.reader import LifFile
from tqdm import tqdm
import os
import cv2

# def bin2x2(image):
#     """Bin a 2D numpy array by 2x2 averaging."""
#     h, w = image.shape
#     h2, w2 = h // 2, w // 2
#     return image[:h2*2, :w2*2].reshape(h2, 2, w2, 2).mean(axis=(1, 3))

# === Helper: 2x2 binning ===
def bin2x2(image):
    h, w = image.shape
    h2, w2 = (h // 2) * 2, (w // 2) * 2
    img = image[:h2, :w2].astype(np.float32)
    top_left     = img[0::2, 0::2]
    top_right    = img[0::2, 1::2]
    bottom_left  = img[1::2, 0::2]
    bottom_right = img[1::2, 1::2]
    return (top_left + top_right + bottom_left + bottom_right) / 4.0

# === Helper: compute one static Gaussian background from the whole stack ===
def compute_static_background(lif_image, n_frames, sample_stride=20, sigma=50):
    """Compute one smooth illumination profile from a subset of frames."""
    frames = []
    for t in range(0, n_frames, sample_stride):
        try:
            frame = np.array(lif_image.get_frame(z=0, t=t), dtype=np.float32)
            frames.append(frame)
        except Exception as e:
            print(f"⚠️ Skipping frame {t}: {e}")
            continue
    if not frames:
        raise ValueError("No frames loaded for background computation.")
    mean_image = np.mean(frames, axis=0)
    background = cv2.GaussianBlur(mean_image, (0, 0), sigma)
    background[background == 0] = 1e-6  # avoid divide-by-zero
    return background

# === Helper: apply flat-field correction ===
def apply_static_flatfield(frame, background):
    frame = frame.astype(np.float32)
    corrected = frame / background
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    return corrected.astype(np.uint8)

# === Main function ===
def LIFbin(lif_path="/Users/mikeodonnell/Desktop/pumping_test/LIfExport/2025-11-10_WT_gcamp8f_off_food.lif"):
    output_dir = os.path.splitext(lif_path)[0] + "_exported"
    fps = 30
    os.makedirs(output_dir, exist_ok=True)

    lif = LifFile(lif_path)

    for series_index, series in enumerate(lif.image_list):
        name = series.get("name", f"series_{series_index+1}")
        dims = series["dims"]
        n_frames = dims.t  # e.g., 900
        print(f"Processing series {series_index+1}/{len(lif.image_list)}: {name} ({n_frames} frames)")

        lif_image = lif.get_image(series_index)

        # === Compute static background once per series ===
        print("Computing static Gaussian background...")
        background = compute_static_background(lif_image, n_frames, sample_stride=20, sigma=50)

        out_path = os.path.join(output_dir, f"{series_index+1:02d}_{name}.avi")
        writer = imageio.get_writer(
            out_path,
            fps=fps,
            format="FFMPEG",
            codec="rawvideo",
            output_params=["-pix_fmt", "gray"]
        )

        for t in tqdm(range(n_frames), desc=f"Series {series_index+1}"):
            try:
                frame = np.array(lif_image.get_frame(z=0, t=t), dtype=np.uint8)
            except Exception as e:
                print(f"⚠️ Error reading frame {t}: {e}")
                continue

            # # === Apply precomputed flat-field correction ===
            # frame_corr = apply_static_flatfield(frame, background)

            # === Bin 2x2 ===
            frame_binned = bin2x2(frame)
            frame_binned = np.round(frame_binned).astype(np.uint8)

            writer.append_data(frame_binned)

        writer.close()
        print(f"✅ Saved {out_path}")

    print("🎉 All series processed successfully.")