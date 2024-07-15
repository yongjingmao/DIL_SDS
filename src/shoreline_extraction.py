import sys
#sys.path.append('/home/z3541792/Github/SatelliteCloudGenerator/')

import rasterio as rio
from rasterio.plot import show
from pathlib import Path
import os
import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cloud_simulate import add_cloud
sys.path.append('../../CoastSat')
from coastsat.SDS_shoreline import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, 
                        default='data/Narrabeen/S1_Landsat', 
                        help='Data path for optical images', metavar='')
                        
    

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args
    
    
def retrieve_SDS(img, mask, crs, georef, min_length, ref_shoreline=None, max_dist_ref=150):
    #S2_mask = np.zeros([S2_img.shape[0], S2_img.shape[1]])==1
    settings = {
        'output_epsg':crs,
        'min_length_sl':min_length,
        'dist_clouds': 30,
    }    
    if ref_shoreline is not None:
        settings['reference_shoreline'] = ref_shoreline
        settings['max_dist_ref'] = 150
    im_ref_buffer = create_shoreline_buffer(mask.shape, georef, crs,
                                        10, settings)
    
    if np.sum(mask==0)>5000: # Clear pixels above 5000
        contours_mwi, t_mndwi = find_wl_contours1(img, mask, im_ref_buffer)
        shoreline = process_shoreline(contours_mwi, mask, np.zeros(mask.shape), georef, crs, settings)
    else: # To few clear pixels to derive shoreline position
        shoreline = np.array([[],[]]).transpose()
    return shoreline
    
def crop_and_resize(img, resize, profile):
    """
    Crop and resize img, keeping relative ratio unchanged
    """
    img = img.transpose(1, 2, 0).astype('float32')
    h, w = img.shape[:2]
    source = 1. * h / w
    target = 1. * resize[0] / resize[1]
    if source >= target:
        margin = int((h - w * target) // 2)
        img = img[margin:h-margin]
        cols_left, cols_right = 0, w
        rows_top, rows_bottom = margin, h-margin

    elif source < target:
        margin = int((w - h / target) // 2)
        img = img[:, margin:w-margin]
        cols_left, cols_right = margin, w-margin
        rows_top, rows_bottom = 0, h

    left, top = rasterio.transform.xy(profile['transform'], rows_top-0.5, cols_left-0.5)
    right, bottom = rasterio.transform.xy(profile['transform'], rows_bottom-0.5, cols_right-0.5)
    bbox = BoundingBox(left=left, bottom=bottom, right=right, top=top)
    transform=rasterio.transform.from_bounds(*bbox, width=resize[1], height=resize[0])

    img = cv2.resize(img, (resize[1], resize[0]), interpolation=cv2.INTER_LINEAR)

    if len(img.shape)!=3:
        img = np.expand_dims(img, axis=-1)

    img = img.transpose(2, 0, 1).astype(np.float32)
    return img, transform    
 
 def polygon_to_binary_image(polygon, src):    
    """
    Convert polygon to binary images
    """
    # Open the .tif file
    image_shape = src.shape
    # Get the transform
    transform = src.transform
    
    # Rasterize the polygon
    binary_image = rasterize(
        [(polygon, 1)],          # List of (geometry, value) pairs
        out_shape=image_shape,   # Shape of the output binary image
        transform=transform,     # Transform to map the polygon to the image grid
        fill=0,                  # Fill value for areas outside the polygon
        dtype='float32'            # Data type of the output array
    )

    return np.expand_dims(binary_image, 0)
    
