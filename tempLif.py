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
from ODLabTracker import tracking, filePreprocess

####### 1. Setup #############
import sys
import argparse

# ## Using argparse (more robust for complex arguments)
# parser = argparse.ArgumentParser(description="Process a file.")
# #parser.add_argument("-i", "--interactive", action="store_true", help="interactive file selection")
# parser.add_argument("-f", "--filename", help="The path to the input file, if specified")
# parser.add_argument("-c", "--config", help="Configuration (yaml) file, if specfified")
# parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
# args = parser.parse_args()

# if args.filename:
#     print("batch mode")
#     print(f"Processing file: {os.path.join(os.getcwd(),args.filename)}")
#     file_path = os.path.join(os.getcwd(),args.filename)
# else:
#     import tkinter as tk
#     from tkinter import filedialog
    
#     root = tk.Tk()
#     root.withdraw()
    
#     file_path = filedialog.askopenfilename(title="Select a Video File")
    
#     root.destroy()

# result_path = os.path.join(f"{os.path.splitext(file_path)[0]}_results")
# print(f"results will be saved to {result_path}")
# os.mkdir(result_path)

# if args.verbose:
#     print("Verbose mode enabled.")

# #### config file setup ####
# if args.config:
#     config_path = os.path.join(os.getcwd(),args.config)
#     import yaml

#     with open(config_path, 'r') as f:
#         config_data = yaml.safe_load(f)
    
#     min_area = config_data['min_area'] #min area of worm in pixels
#     max_area = config_data['max_area'] #max area of worm in pixels
#     gap_range = config_data['gap_range'] # max number of frame gap to link worms
#     if config_data['thresh'] == 'None':
#         thresh = None
#     else:
#         thresh = config_data['thresh'] # manual threshold - use if too few worms detected or too many short tracks
#     search_range = config_data['search_range'] # max pixel distance to link tracks across frames
#     min_length = config_data['min_length'] # minimum length of track in frames to keep
#     frame_rate = config_data['frame_rate'] # FPS - only necessary for speed analysis
#     illumination = config_data['illumination'] # illumination source, 0 = white worms on dark (e.g. IR), 1 = dark worms on light
#     subsample = config_data['subsample']
    
# else:
#     #default values small plate on IR light:
#     min_area = 200 #min area of worm in pixels
#     max_area = 2000 #max area of worm in pixels
#     gap_range = 8 # max number of frame gap to link worms
#     thresh = None # manual threshold - use if too few worms detected or too many short tracks
#     search_range = 60 # max pixel distance to link tracks across frames
#     min_length = 25 # minimum length of track in frames to keep
#     frame_rate = 10 # FPS - only necessary for speed analysis
#     illumination = 0 # illumination source, 0 = white worms on dark (e.g. IR), 1 = dark worms on light

filePreprocess.LIFbin()
