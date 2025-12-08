#!/usr/bin/env python
# coding: utf-8

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
import yaml
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

#from IPython.display import Video, Markdown, display

import ODLabTracker
from ODLabTracker import tracking, filePreprocess, batchtools

####### 1. Setup #############
import sys
import argparse

## Using argparse (more robust for complex arguments)
parser = argparse.ArgumentParser(description="Process a file.")
#parser.add_argument("-i", "--interactive", action="store_true", help="interactive file selection")
parser.add_argument("-f", "--filename", help="The path to the input file, if specified")
parser.add_argument("-c", "--config", help="Configuration (yaml) file, if specfified")
parser.add_argument("-l", "--lif", action="store_true", help="Process LIF multi-video file")
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
args = parser.parse_args()

if args.lif:
    print("exporting video files from LIF file")
    if args.filename:
        lif_path = args.filename
    else:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
    
        lif_path = filedialog.askopenfilename(title="Select a LIF file")

        root.destroy()
    
    filePreprocess.LIFbin(lif_path = lif_path)
    export_path = os.path.join(f"{os.path.splitext(lif_path)[0]}_exported")
    print(f"✅ exported videos to {export_path}")
    # os.makedirs(export_path, exist_ok=True)

    print(f"Using config file {args.config}")

    #### config file setup ####
    config_path = os.path.join(os.getcwd(),args.config)
    import yaml

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
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
        
    print('PARAMETER SETTINGS:')
    print(f'--minimum area of worm in pixels: {min_area}')
    print(f'--maximum area of worm in pixels: {max_area}')
    print(f'--gap range of worms in frames: {gap_range}')
    if thresh is None:
        print(f'--automatically calculating threshold')
    else:
        print(f'--manual threshold: {thresh}')
    print(f'--maximum pixel distance to link worms: {search_range}')
    print(f'--minimum length of worm track to keep in frames: {min_length}')
    print(f'--frame rate to use for speed analysis: {frame_rate}')
    if illumination == 0:
        print(f'--analyzing light worms on dark background')
    else:
        print(f'--analyzing dark worms on light background')
    if subsample > 1:
        print(f"--keeping one out of every {subsample} frames")

    videos = []
    video_extension = ".avi"
    for filename in os.listdir(export_path):
        if os.path.isfile(os.path.join(export_path, filename)): # Ensure it's a file
            _, extension = os.path.splitext(filename)
            if extension.lower() == video_extension.lower(): # Case-insensitive comparison
                videos.append(filename)

    for i in videos:
        current_file = f"{os.path.join(export_path,i)}"
        # print(f"Processing video: {current_file}")
        batchtools.FastTrackIter(file_path = current_file, config_data = config_data)
