# importing, data handling
import numpy as np
import pandas as pd
import ipyfilechooser
import os
import imageio
import imageio.v3 as iio
import tifffile
from PIL import Image
from skimage.color import rgb2gray
import time

# plotting
import matplotlib  as mpl
import cv2
import matplotlib.pyplot as plt

# thresholding
from skimage import io, color, filters, morphology, measure
from skimage.draw import rectangle_perimeter

# tracking
import trackpy as tp

from IPython.display import Video



def preprocess_frame(img, min_area, max_area, thresh, illumination):
        # Convert PIL.Image or other inputs to numpy
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    gray = img

    # Use provided threshold or compute new one
    if thresh is None:
        print('auto threshold')
        thresh = filters.threshold_otsu(gray)

    if illumination == 0:
        bw = gray > thresh   # worms lighter
    else:
        bw = gray < thresh   # worms darker

    bw = morphology.remove_small_objects(bw, min_size=min_area)

    labeled = measure.label(bw)
    props = measure.regionprops(labeled)
    areas = measure.regionprops_table(labeled, properties = ['area_convex'])

    mask = np.zeros_like(bw, dtype=bool)
    for prop in props:
        if min_area <= prop.area_convex <= max_area:
            mask[labeled == prop.label] = True

    return mask, props, thresh

def draw_boxes_labels(frame, props):
    overlay = np.array(frame).copy()

    # Convert RGB to grayscale [0-255]
    if overlay.ndim == 3:
        overlay = color.rgb2gray(overlay)
        overlay = (overlay * 255).astype(np.uint8)

    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        rr, cc = rectangle_perimeter((minr, minc), end=(maxr, maxc), shape=overlay.shape[:2])
        overlay[rr, cc] = 255
    return overlay

def show_background(input_video=None,
                   input_file=None,
                   first_frame=None,
                   num_frames=None):

    if first_frame.ndim == 3 and first_frame.shape[-1] == 3:
        first_frame = rgb2gray(first_frame)
        first_frame = (first_frame * 255).astype(np.uint8)
    elif first_frame.ndim == 2:
        first_frame = first_frame.astype(np.uint8)

    ### check for uneven luminance ####
    background = filters.gaussian(first_frame, sigma=50, preserve_range=True)
    backsubbed = (first_frame - background)

    # convert background subtracted image back to uint8 (0-255 integers):
    min_val = np.min(backsubbed)
    max_val = np.max(backsubbed)
    if max_val - min_val == 0:
        # Handle case of a flat image to avoid division by zero
        backsubbed = np.zeros_like(image_float, dtype=np.uint8)
    else:
        # normalize across the range of values in the subtracted image:
        backsubbed = (backsubbed - min_val) / (max_val - min_val)
        
        # Scale to 0-255 and convert to uint8
        backsubbed = np.round(backsubbed * 255).astype(np.uint8)
        # backsubbed = filters.gaussian(backsubbed, sigma=5, preserve_range=True)
    
    background_diag = np.diag(background)
    background_invdiag = np.diag(np.fliplr(background))
    backsubbed_diag = np.diag(backsubbed)
    x = np.arange(len(background_diag))

    coefficients_diag = np.polyfit(x, background_diag, 1)
    coefficients_invdiag = np.polyfit(x, background_invdiag, 1)

    print(f"Slope of background top-left to bottom right: {coefficients_diag[0]}")
    print(f"Slope of background bottom-left to top right: {coefficients_invdiag[0]}")
    

    if np.abs(coefficients_diag[0]) or np.abs(coefficients_invdiag[0]) > 0.02:
        print("Significantly uneven background luminance, need to subtract background")
        backsub = True
        f = plt.figure(figsize = (6,12))
        f.add_subplot(3,1,1)
        plt.imshow(background, cmap="gray")
        plt.title(f"Background luminance original and corrected")
        f.add_subplot(3,1,2)
        plt.imshow(first_frame, cmap="gray")
        f.add_subplot(3,1,3)
        plt.imshow(backsubbed, cmap="gray")

        plt.figure(figsize=(10, 6))
        plt.plot(x,background_diag)
        plt.plot(x,background_invdiag)
        plt.plot(x,backsubbed_diag)
        # imiter_vid = input_video # should be in imiter format
        print(f"returning background image of type {type(background)}")
        print(f"background subtracted image is type {type(backsubbed)}")

    return(background, backsub)

def convert_8bit(frame):
    if frame.ndim == 3 and frame.shape[-1] == 3:
        frame = rgb2gray(frame)
        frame = (frame * 255).astype(np.uint8)
    elif frame.ndim == 2:
        frame = frame.astype(np.uint8)
    return(frame)

def float_to8bit(frame):
    # convert background subtracted or averaged image back to uint8 (0-255 integers):
    min_val = np.min(frame)
    max_val = np.max(frame)
    if max_val - min_val == 0:
        # Handle case of a flat image to avoid division by zero
        frame = np.zeros_like(frame, dtype=np.uint8)
    else:
        # normalize across the range of values in the subtracted image:
        frame = (frame - min_val) / (max_val - min_val)
        
        # Scale to 0-255 and convert to uint8
        frame = np.round(frame * 255).astype(np.uint8)
        # print(np.max(frame), np.min(frame))
    return(frame)

def subtract_background(frame,
                       average_frame):
    backsubbed = frame.astype(np.float32) / (average_frame.astype(np.float32) + 1)
    # print(f"max value in avg frame is: {np.max(average_frame)}, min is {np.min(average_frame)}")
    frame = float_to8bit(backsubbed)
    return(frame)

def process_video(min_area, 
    max_area, 
    thresh,
    input_video=None, # should be in imiter format
    output_path=None, 
    max_frames=None,
    fps=None, 
    save_as="mp4",
    illumination=0,
    subsample=None,
    backsub=None,
    background=None, 
    average_frame=None):
    """
    Pre-Process TIFF stack or video, to preview the annotation of worms, and optimize settings. Save video with objects detected as MP4 or TIFF stack.

    Parameters
    ----------
    input_video : imiter object
        Video read in via imiter(vid) - imageio v3. 
    output_path : str or None
        Path to save annotated output (mp4 or tif). If None, nothing saved.
    max_frames : int or None
        Process only first N frames (for testing).
    save_as : str
        "mp4" or "tif"
    subsample : Frames to subsample. Video length will be n/subsample
    """
    os.makedirs(output_path,exist_ok=True)

    import time

    frames = []
    # ----- convert to grayscale 8-bit ---------
    with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
        start_time = time.time()
        for i, frame in enumerate(input_video):
            if i % subsample == 0: # this is zero only for multiples of subsample
                if i / subsample <= max_frames: # read in only first 50 frames of subsample
                    if frame.ndim == 3 and frame.shape[-1] == 3:
                        if i == 1:
                            print("converting to grayscale images")
                        frame = rgb2gray(frame)
                        frame = (frame * 255).astype(np.uint8)
                    elif frame.ndim == 2:
                        if i == 1: 
                            print("already grayscale, converting to 8-bit")
                        frame = frame.astype(np.uint8)
                    if backsub == True:
                        subtracted = subtract_background(frame = frame, average_frame = average_frame)
                        if i == 0:
                            thresh = np.percentile(subtracted, .5)
                        frame = subtracted
                    # Append the processed frame to the list
                    frames.append(frame)
        end_time = time.time()
    print(f"Converting frames using imiter of video took {end_time - start_time} seconds")

    # if backsub == True:
    #     backsubbed = []
    #     print("subtracting background")
    #     for i, frame in enumerate(frames):
    #         backsub_img = (frame - background)
    #         # convert background subtracted image back to uint8 (0-255 integers):
    #         min_val = np.min(backsub_img)
    #         max_val = np.max(backsub_img)
    #         if max_val - min_val == 0:
    #             # Handle case of a flat image to avoid division by zero
    #             backsub_img = np.zeros_like(image_float, dtype=np.uint8)
    #         else:
    #             # normalize across the range of values in the subtracted image:
    #             backsub_img = (backsub_img - min_val) / (max_val - min_val)
            
    #         # Scale to 0-255 and convert to uint8
    #         backsub_img = np.round(backsub_img * 255).astype(np.uint8)
    #         backsubbed.append(backsub_img)
    #     frames = backsubbed
        
    
    first_frame = frames[0]

    # result_path = os.path.join(f"{os.path.splitext(input_path)[0]}_results")
    print(f"results will be saved to {output_path}")
    workingDir = os.getcwd()
    
    if save_as == "mp4" and output_path:
        writer = imageio.get_writer(os.path.join(output_path,"worms_annotated.mp4"), fps=fps)
    else:
        writer = None

    overlays = []  # store frames if saving as TIFF

    # --- Get threshold from first frame ---
    if thresh is None:
        mask1, props1, global_thresh = preprocess_frame(first_frame, 
                                                        min_area, 
                                                        max_area, 
                                                        thresh, 
                                                        illumination=illumination)
        print("Auto calculated threshold value is",global_thresh)
 
    else:
        global_thresh = thresh
        print("Manual threshold value is",global_thresh)
        mask1, props1, _ = preprocess_frame(first_frame,
                                            min_area,
                                            max_area,
                                            thresh=global_thresh,
                                            illumination=illumination)
        
    # ------ plot first frame mask with histogram of object sizes ------
    areas = []
    for prop in props1:
        area = prop.area
        areas.append(area)
    print('Mean area of objects in pixels is', np.mean(areas))
    if (np.mean(areas) < min_area) | (np.mean(areas) > max_area):
        print("which is outside of your min and max area estimate")
    f = plt.figure(figsize=(6,4))
    ax = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    ax.imshow(mask1, cmap="gray")
    ax.set_title(f"Frame 1, worms: {len(props1)}")
    # ax.axis("off")
    ax2.hist(areas, bins = 10)
    ax2.set_title("Areas of detected objects")
    ax.set_xlabel("Area")
    ax.set_ylabel("Number of objects")
    plt.tight_layout()
    plt.show()

    all_areas = []
    for i, frame in enumerate(frames):
        if max_frames and i >= max_frames:
            break

        mask, props, _ = preprocess_frame(frame, 
                                          min_area, 
                                          max_area, 
                                          thresh=global_thresh,
                                          illumination=illumination)
                                    # Debug: show first frame

        overlay = draw_boxes_labels_gray(frame, props)
        frame_areas = [prop.area for prop in props]
        all_areas.extend(frame_areas)

        if save_as == "mp4" and writer:
            writer.append_data(overlay)
        elif save_as == "tif":
            overlays.append(overlay)
    print(len(all_areas))
    
    f2 = plt.figure(figsize=(6,4))
    plt.hist(all_areas, bins = 20)
    plt.title(f"{len(all_areas)} objects detected (unfiltered, 50 frames)")
    plt.show()
    
    if save_as == "mp4" and writer:
        writer.close()
    elif save_as == "tif" and output_path:
        tifffile.imwrite(os.path.join(output_path,"worms_annotated.tif"), np.array(overlays), photometric="minisblack")
    return global_thresh

def draw_boxes_labels_gray(frame, props):
    """
    Draw bounding boxes + worm labels on grayscale frame.
    """
    overlay = np.array(frame).copy()
    # Convert RGB to grayscale [0-255] - already done in process_video
    # if overlay.ndim == 3:
    #     from skimage import color
    #     overlay = color.rgb2gray(overlay)
    #     overlay = (overlay * 255).astype(np.uint8)

    for prop in props:
        # if no objects are detected generate error:
        if (len(prop.bbox)) == 4:
            minr, minc, maxr, maxc = prop.bbox
            rr, cc = rectangle_perimeter(
                (minr, minc), end=(maxr, maxc), shape=overlay.shape[:2]
            )
            overlay[rr, cc] = 255  # white box

            # Add label text (use regionprops label)
            y, x = prop.centroid
            cv2.putText(
                overlay, str(prop.label),
                (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,), 1, cv2.LINE_AA
            )
        else:
            print("error - too many params")
    return overlay

def collect_detections(stack, global_thresh, min_area, max_area, illumination=0):
    # this is just a wrapper for preprocess_frame to track centroids
    frames = stack
    records = []
    # Convert to grayscale only if RGB
    for frame_no, frame in enumerate(frames):
        # print(f"keeping frame {i}")
        print(f"\rThresholding and getting centroids for frame: {frame_no}", end="", flush=True)
        time.sleep(0.0001)
        mask, props, _ = preprocess_frame(frame,
                                          min_area=min_area,
                                          max_area=max_area,
                                          thresh=global_thresh,
                                          illumination=illumination)
        for prop in props:
            if prop.area_convex < min_area or prop.area_convex > max_area:
                print(f"area: {prop.area_convex} object outside area range")
                prop = None
            else:
                y, x = prop.centroid  # note (row, col) = (y, x)
                records.append({
                    "frame": frame_no,
                    "x": x,
                    "y": y,
                    "area": prop.area,
                    "major_axis": prop.axis_major_length,
                    "minor_axis": prop.axis_minor_length,
                    "orientation": prop.orientation,
                    "eccentricity": prop.eccentricity,
                    "euler_num": prop.euler_number,
                    "solidity": prop.solidity,
                    "area_convex": prop.area_convex
                })
    return pd.DataFrame(records)

def link_tracks(detections, search_range=50, memory=3, quiet=False):
    """
    Link worm detections into tracks.
    search_range: max distance a worm can move between frames
    memory: how many frames a worm can vanish and still be linked
    """
    print("Linking tracks")
    # Suppress informational messages from trackpy so output isn't crashed
    if quiet:
        tp.quiet()
    linked = tp.link_df(detections, search_range=search_range, memory=memory)
    return linked

def draw_tracks_gray(frame, props, tracks, frame_no):
    overlay = np.array(frame).copy() # should inherit frames from above, no need to convert again
    if overlay.shape[-1] == 3:
        overlay = color.rgb2gray(overlay)
        overlay = (overlay * 255).astype(np.uint8)
    # Get track IDs for this frame
    frame_tracks = tracks[tracks["frame"] == frame_no]

    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        rr, cc = rectangle_perimeter((minr, minc), end=(maxr, maxc), shape=overlay.shape[:2])
        overlay[rr, cc] = 255

        # Match centroid to track
        y, x = prop.centroid
        match = frame_tracks[(frame_tracks["x"].sub(x).abs() < 1) &
                             (frame_tracks["y"].sub(y).abs() < 1)]
        if not match.empty:
            track_id = int(match["particle"].iloc[0])
            cv2.putText(
                overlay, str(track_id),
                (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,), 1, cv2.LINE_AA
            )
    return overlay

def filter_short_tracks(tracks, min_length=10):
    """
    Remove worms (particles) with fewer than `min_length` frames.
    """
    counts = tracks.groupby("particle")["frame"].count()
    keep_ids = counts[counts >= min_length].index
    return tracks[tracks["particle"].isin(keep_ids)].copy()

def filter_area_change(tracks, max_area_cv = 0.2):
    """
    Remove worms (particles) with fewer than `min_length` frames.
    """
    
    grouped_tracks = tracks.groupby("particle")
    # mean_area = grouped_tracks['area'].agg(
    # mean_area = grouped_tracks[)
    area_variation = grouped_tracks['area'].agg(
    cv=lambda x: np.std(x) / np.mean(x) if np.mean(x) > 0 else 0)

    # Set a maximum allowed coefficient of variation for the area
    max_area_cv = max_area_cv # Example: allow up to a 10% change relative to the mean

    # Get the IDs of the particles to keep
    particles_to_keep = area_variation[area_variation['cv'] <= max_area_cv].index

    # Filter the original trajectories DataFrame
    stable_trajectories = tracks[tracks['particle'].isin(particles_to_keep)]
    return stable_trajectories

def plot_trajectories(stack, tracks, output_path, background="first"):
    """
    Plot worm trajectories on top of an image.

    Parameters
    ----------
    stack : ndarray
        TIFF stack (frames, h, w)
    tracks : DataFrame
        trackpy-linked detections with columns [frame, x, y, particle]
    background : str
        "first" = use first frame
        "mean" = use mean intensity projection
    """
    bg = stack

    #    bg = np.zeros_like(stack[0])

    plt.figure(figsize=(10,8))
    plt.imshow(bg, cmap="gray")


    # Plot trajectories for each worm
    for pid, worm in tracks.groupby("particle"):
        plt.plot(worm["x"], worm["y"], marker=".", markersize=1, label=f"Worm {pid}")
    plt.legend(loc="upper right", fontsize=6)
    plt.title("Worm Trajectories")
    plt.savefig(os.path.join(output_path,"trackPlot.png"))
    plt.show()

def load_video_frames(path, max_frames=None):
    """
    Load frames from AVI or TIFF.
    Returns a numpy array: (n_frames, h, w), grayscale.
    """
    frames = []

    # imageio can open both .avi and .tif
    reader = iio.imiter(path)
    for i, frame in enumerate(reader):
        if max_frames and i >= max_frames:
            break
        # If RGB, convert to grayscale
        if frame.ndim == 3:
            frame = rgb2gray(frame)
            frame = (frame * 255).astype(np.uint8)
        frames.append(frame)

    return np.stack(frames, axis=0)
