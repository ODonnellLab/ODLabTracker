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
import ODLabTracker
from ODLabTracker import tracking

def FastTrackIter(file_path,config_data):
    ####### 1. Setup #############
    import sys
    import os

    filename = os.path.splitext(file_path)[0]
    filename_short = os.path.basename(filename)
    print(f'Processing file: {filename_short}')
    # file_path = filename

    result_path = os.path.join(f"{filename}_results")
    print(f"results will be saved to {result_path}")
    os.makedirs(result_path, exist_ok=True)

    min_area = config_data['min_area'] #min area of worm in pixels
    max_area = config_data['max_area'] #max area of worm in pixels
    gap_range = config_data['gap_range'] # max number of frame gap to link worms
    if config_data['thresh'] == 'None':
        thresh = None
    else:
        thresh = config_data['thresh'] # manual threshold - use if too few worms detected or too many short tracks
    search_range = config_data['search_range'] # max pixel distance to link tracks across frames
    min_length = config_data['min_length'] # minimum length of track in frames to keep
    frame_rate = config_data['frame_rate'] # FPS - only necessary for speed analysis
    illumination = config_data['illumination'] # illumination source, 0 = white worms on dark (e.g. IR), 1 = dark worms on light
    subsample = config_data['subsample']

    # ###### 2. warnings for non-optimal video ######
    import time

    start_time = time.time()
    first_frame = iio.imread(file_path, index = 0,  plugin = "pyav")
    imiter_vid = iio.imiter(file_path, plugin = "pyav")
    image_props = iio.improps(file_path, plugin="pyav")
    print(image_props)
    num_frames = image_props.shape[0]

    if first_frame.shape[-1] == 3:
        print("\n!!!Video is RGB, need to convert to grayscale 8-bit - consider changing video output to this type ahead of time!!!\n")

    frames = []

    ##### 3. tracking #######
    if subsample > 1:
        print(f'Running tracking on {num_frames/subsample} frames from the original {num_frames} frames')
    else:
        print(f'Running full tracking on {num_frames} frames')

    # not sure why matmul errors happen for the first rgb2gray call but suppress this block:
    with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
        start_time = time.time()
        for i, frame in enumerate(imiter_vid):
            if i % subsample == 0:
                # print(f"keeping frame {i}")
                print(f"\rKeeping frame: {i}", end="", flush=True)
                time.sleep(0.001)
                if frame.ndim == 3 and frame.shape[-1] == 3:
                    if i/subsample == 1:
                        print("converting to grayscale images")
                    frame = rgb2gray(frame)
                    frame = (frame * 255).astype(np.uint8)
                elif frame.ndim == 2:
                    if i/subsample == 1: 
                        print("already grayscale, converting to 8-bit")
                    frame = frame.astype(np.uint8)
                # Append the processed frame to the list
                frames.append(frame)
        end_time = time.time()
        print(f"  Reading in {len(frames)} frames from the full length video took {end_time - start_time} seconds")

    # now simple track for the whole video and link tracks together 
    # using new first frame after subsampling
    first_frame = frames[0]
    # # now simple track for the whole video and link tracks together
    print(f"Tracking and linking objects from {len(frames)} frames")

    # Compute global threshold
    max_objects = config_data.get('max_objects', None)

    if thresh is None:
        _, _, global_thresh = tracking.preprocess_frame(first_frame,
                                                        min_area,
                                                        max_area,
                                                        thresh,
                                                        illumination=illumination,
                                                        max_objects=max_objects)
    else:
        print("tracking video using manual threshold")
        global_thresh = thresh
        
    # Collect detections
    detections = tracking.collect_detections(frames, 
                                            global_thresh = global_thresh, 
                                            min_area=min_area, 
                                            max_area=max_area,
                                            illumination=illumination)

    # Link tracks
    # Suppress all trackpy logging messages
    # tp.ignore_logging()

    tracks = tracking.link_tracks(detections, search_range=search_range, memory=gap_range, quiet = True)

    # filtering tracks whose object area changes more than some percentage%

    print("calculating speed")
    window_size = 2
    tracks['dx'] = tracks.groupby('particle')['x'].diff()
    tracks['dy'] = tracks.groupby('particle')['y'].diff()
    tracks['dt'] = tracks.groupby('particle')['frame'].diff()
    # Calculate instantaneous speed (distance / time) of the centroid (not ideal) This is per frame (not per sec)
    tracks['speed_int'] = np.sqrt(tracks['dx']**2 + tracks['dy']**2) 
    tracks['speed_roll'] = tracks.groupby('particle')['speed_int'].rolling(
        window = window_size,
        min_periods=1).mean().reset_index(level=0, drop=True)
    tracks['ecc_roll'] = tracks.groupby('particle')['eccentricity'].rolling(
        window = window_size,
        min_periods=1).mean().reset_index(level=0, drop=True)
    # calculate the area of a rectangle bound by the major and minor axes
    tracks['area_rect'] = tracks['major_axis'] * tracks['minor_axis']
    tracks['area_roll'] = tracks.groupby('particle')['area_rect'].rolling(
        window = window_size,
        min_periods=1).mean().reset_index(level=0, drop=True)
    tracks['angvel'] = tracks.groupby('particle')['orientation'].diff()
    # Handle angle wrapping (e.g., crossing the -pi to pi boundary)
    tracks['angvel'] = np.abs(np.arctan2(np.sin(tracks['angvel']), np.cos(tracks['angvel'])))
    tracks['angvel_roll'] = tracks.groupby('particle')['angvel'].rolling(
        window = window_size,
        min_periods=1).mean().reset_index(level=0, drop=True)

    # guess at when the animal is turning based on speed and eccentricity
    # print(np.nanmax(tracks['speed']))
    # print(np.nanpercentile(tracks['speed'], 75))
    # print(np.nanpercentile(tracks['speed'], 25))

    # Now tracks['particle'] is a persistent worm ID

    # visualize the track length of each "worm"
    # if there are background particles, this will lead to a ton of short tracks

    #### summarize some features after filtering ##### 
    counts = tracks.groupby("particle")["frame"].count()
    print('Mean track length is ',np.ceil(np.mean(counts)/2), ' frames')
    print('Minimum track length is ',int(min(counts)))
    print('Maximum track length is ',int(max(counts)))
    #plt.figure(figsize=(10,8))
    binwidth = 25
    plt.hist(counts, bins=range(int(min(counts)), int(max(counts)) + binwidth, binwidth))
    plt.xlabel("length of track in frames")
    plt.title("histogram of worm track lengths")
    plt.xlabel("length of track in frames \nif too many short tracks, try increasing gap_range, \nif your real worms are disconnected, \nor increase threshold if there are too many small objects")
    plt.ylabel("number of worm tracks")
    plt.show()

    # Remove short tracks
    print(f'removing tracks shorter than {min_length} frames')
    tracks = tracking.filter_short_tracks(tracks, min_length=min_length)

    # # filtering tracks whose object area changes more than 10%
    # tracks = tracking.filter_area_change(tracks, max_area_cv = 0.2)

    # Plot over first frame
    print('plotting linked and filtered worm tracks')
    tracking.plot_trajectories(stack=first_frame, tracks=tracks, output_path=result_path)

    # Save the track centroids to a csv file
    print(f'saving tracked centroids to {os.path.join(result_path,"tracks.csv")}')
    tracks.to_csv(os.path.join(result_path,"tracks.csv"), index=False)