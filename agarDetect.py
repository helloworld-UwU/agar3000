# -*- coding: utf-8 -*-
"""
Created on Thu May 22 11:18:45 2025

@author: Admin
"""

# %%

import sys
import os
sys.path.append( os.path.join(os.getcwd(), "CFUcounter"))

# %%
# Imports
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import cv2
import numpy as np

# %%

# Paths
model_path = os.path.join(os.getcwd(), 'CFUCounter/agar_cfg20221010T2320/mask_rcnn_agar_cfg_0004.h5')
data_path = 'images'

#input_path = sys.argv[1]


# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "my_cfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 2048
    IMAGE_MIN_SCALE = 0
    
    DETECTION_MAX_INSTANCES = 500
    POST_NMS_ROIS_INFERENCE = 8000
    DETECTION_MIN_CONFIDENCE = 0.90

# draw an image with detected objects
def draw_image_with_boxes(filename, boxes_list):
     # load the image
     data = pyplot.imread(filename)
     # plot the image
     pyplot.imshow(data)
     # get the context for drawing boxes
     ax = pyplot.gca()
     # plot each box
     for box in boxes_list:
          # get coordinates
          y1, x1, y2, x2 = box
          # calculate width and height of the box
          width, height = x2 - x1, y2 - y1
          # create the shape
          rect = Rectangle((x1, y1), width, height, fill=False, color='red')
          # draw the box
          ax.add_patch(rect)
     # show the plot
     num_colonies = len(boxes_list)
     print('Number of colonies: ', num_colonies)
     pyplot.axis('off')
     pyplot.show()

# %% 
# define the model
rcnn = MaskRCNN(mode='inference', model_dir='./', config=PredictionConfig())

# %% 
# load coco model weights
rcnn.load_weights(model_path, by_name=True)
# %%
def detectResults(im_path):
    # Load image using Keras
    img = load_img(im_path)
    img = img_to_array(img).astype(np.uint8)

    # Apply Median Blur (remove salt-and-pepper noise)
    #img = cv2.medianBlur(img, 9)

    # Apply Bilateral Filter (smooth while preserving edges)
    #img = cv2.bilateralFilter(img, d=13, sigmaColor=75, sigmaSpace=375)

    # Make prediction
    results = rcnn.detect([img], verbose=0)

    return results

