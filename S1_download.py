# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 11:01:56 2023

@author: Yongjing Mao
"""

from GEE_MLR import GEE_funcs
from GEE_MLR.utilities import check_task_status
from GEE_MLR import utilities
from osgeo import gdal
import os
import pathlib
import time
import math
import json
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio import _warp
import ee

import sys
import zipfile
import requests

sys.path.append('..\..\gee_s1_ard\python-api')
import wrapper


    
# ee.Authenticate()
ee.Initialize()

#%%
def download_tif(image, AOI, crs, bands, filepath):

    # for the old version of ee raise an exception
    if int(ee.__version__[-3:]) <= 201:
        raise Exception('CoastSat2.0 and above is not compatible with earthengine-api version below 0.1.201.' +\
                        'Try downloading a previous CoastSat version (1.x).')
    # for the newer versions of ee
    
    else:       
        # crop and download
        download_id = ee.data.getDownloadId({'image': image,
                                             'region': AOI,
                                             'crs':crs,
                                             'filePerBand': False,
                                             'name': 'image',
                                             'bands': bands,
                                             'crs': crs,
                                             'scale': 10,
                                             'format': 'GEO_TIFF'
                                             })
        url = ee.data.makeDownloadUrl(download_id)
        response = requests.get(url)
        if response.ok:
            with open(filepath, 'wb') as fd:
              fd.write(response.content)
        else:
            raise(Exception(json.loads(response.content)['error']['message']))

#%%
def main():
    """
    ==============================
     Read input and set parameters
    ==============================
    """
    cfg = json.load(open(r"..\config\download_config.json", 'r'))
    S1_param = json.load(open(r"..\config\S1Ard_config.json", 'r'))
    
    """
    Define input parameters
    """

    # Name of Input optical missions, choose from L8 (Landsat8) and S2
    START_DATE = cfg["START_DATE"]  # Starting date of analysis
    END_DATE = cfg["END_DATE"]  # Ending date of analysis
    S1_SMOOTH = cfg["S1_SMOOTH"] # Whether smooth S1 images
    OUT_DIR = cfg["OUT_DIR"]
    # Area of interest (AOI) should be saved into a json or shp file
    AOI_PATH = cfg["AOI_PATH"]
    AOI_NAME = cfg["AOI_NAME"]
    OUT_DIR = cfg["OUT_DIR"]
    
    if not os.path.exists(OUT_DIR):
        pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    AOIs = gpd.read_file(AOI_PATH)
    AOI = list(AOIs.loc[AOIs['Name'] == AOI_NAME, 'geometry'])[0]
    AOI = ee.Geometry.Polygon(list(AOI.exterior.coords)).bounds()

                        
    """
    ==========================
     Preprocess SAR images
    ==========================
    """
    
    """
    Define S1_ard parameters
    """
    if S1_SMOOTH:
        S1_param.update([('START_DATE', START_DATE),
                         ('STOP_DATE', END_DATE),
                         ('ROI', AOI),
                         ('DEM', ee.Image('USGS/SRTMGL1_003'))])

        
        S1 = wrapper.s1_preproc(S1_param).select(['VV', 'VH'])
    else:
        S1 = ee.ImageCollection('COPERNICUS/S1_GRD').select(
            [
                'VV',
                'VH']).filter(
            ee.Filter.listContains(
                'transmitterReceiverPolarisation',
                'VV')).filter(
            ee.Filter.listContains(
                'transmitterReceiverPolarisation',
                'VH')).filter(
            ee.Filter.eq(
                'instrumentMode',
                'IW')) .filterBounds(AOI).filterDate(
            START_DATE,
            END_DATE)
                    
    def count_mask(img):
      mask_count = img.select('VV').mask().eq(0).reduceRegion(ee.Reducer.sum(), AOI, 100)
      return img.set({'mask_count':mask_count.get('VV')})

    S1 = S1.map(count_mask).filterMetadata('mask_count', 'equals', 0)
    
    print('{} images to download'.format(S1.size().getInfo()))
    
    crs = S1.first().projection().crs()
 
    S1_ids = S1.aggregate_array('system:index').getInfo()
    S1_times = S1.aggregate_array('system:time_start').getInfo()
    S1_orbits = S1.aggregate_array('relativeOrbitNumber_start').getInfo()
    meta_df = pd.DataFrame({
        'SAR_id': S1_ids,
        'Time': S1_times,
        'Orbits': S1_orbits,
        })
    meta_df.to_csv(os.path.join(OUT_DIR, 'S1.csv'), index=False)
    
    # Image download
    path = os.path.join(OUT_DIR, 'SAR')

    if not os.path.exists(path):
        os.mkdir(path)
        
    for index, row in meta_df.iterrows():
        filename = row['SAR_id']            

        band = ['VV', 'VH']
        
        filepath = os.path.join(path, filename+'.tif')
        if not os.path.exists(filepath):
            image = S1.filterMetadata(
                'system:index', 'equals', row['SAR_id']).select(
                    band).first()                  
            done = False
            count = 0
            while not done:
                try:
                    print('Donwload {}'.format(filename))
                    download_tif(image, AOI, crs,
                                 band, filepath)
                    done = True
                except:
                    print('\nDownload attempt {} failed, trying again'.format(count))
                    count += 1
                    if count > 1:
                        raise Exception('Failed too many times')
                        done = True
                    else:
                        continue
                        
    """
    ==========================
     Warp images
    ==========================
    """

if __name__ == "__main__":
    main()