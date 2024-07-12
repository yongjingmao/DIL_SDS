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


def add_cloudMask_Landsat(img):
    """ Step2: Process Landsat 8 cloud flag
    In the cloud mask, 1 is for cloudy pixel.
    """
    # Bits 3 and 5 are cloud shadow and cloud, respectively.
    dilated_bitMask = 1 << 1
    cirrus_bitMask = 1 << 2
    clouds_bitMask = 1 << 3
    cloudshadow_bitMask = 1 << 4


    # Select band cloud flag
    qa = img.select('QA_PIXEL')

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudshadow_bitMask)\
        .Or(qa.bitwiseAnd(dilated_bitMask))\
        .Or(qa.bitwiseAnd(clouds_bitMask))\
        .Or(qa.bitwiseAnd(cirrus_bitMask)).rename('Mask')

    return img.resample('bicubic').addBands(mask)

def add_cloudMask_S2(img):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability').rename('QA_PIXEL')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(50).rename('Mask')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))

def scale_S2(img):
    return img.multiply(0.0001).copyProperties(
        img).set({'system:time_start':img.get('system:time_start')})

def scale_Landsat(img):
    return img.multiply(2.75e-05).subtract(0.2).copyProperties(
        img).set({'system:time_start':img.get('system:time_start')})

#%%
def main():
    """
    ==============================
     Read input and set parameters
    ==============================
    """
    cfg = json.load(open(r"..\config\download_config.json", 'r'))
    
    """
    Define input parameters
    """

    # Name of Input optical missions, choose from L8 (Landsat8) and S2
    # (Sentinel 2)
    OPTICAL_MISSION = cfg["OPTICAL_MISSION"]
    START_DATE = cfg["START_DATE"]  # Starting date of analysis
    END_DATE = cfg["END_DATE"]  # Ending date of analysis
    PROCESS_STAGE = cfg["PROCESS_STAGE"]
    OUT_DIR = cfg["OUT_DIR"]
    # Area of interest (AOI) should be saved into a json or shp file
    AOI_PATH = cfg["AOI_PATH"]
    AOI_NAME = cfg["AOI_NAME"]
    OUT_DIR = cfg["OUT_DIR"]
    
    if not os.path.exists(OUT_DIR):
        pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    """
    ==========================
     Preprocess optical images
    ==========================
    """
    AOIs = gpd.read_file(AOI_PATH)
    AOI = list(AOIs.loc[AOIs['Name'] == AOI_NAME, 'geometry'])[0]
    AOI = ee.Geometry.Polygon(list(AOI.exterior.coords)).bounds()


    # if OPTICAL_MISSION not in ['L8', 'S2']:
    #     print('Invalid Mission name, please select either S2 or L8')
    #     raise utilities.InvalidInputs

    # Optical image collection
    collection_ids = {
        'SR': {'L8': "LANDSAT/LC08/C02/T1_L2",
               'L9': "LANDSAT/LC09/C02/T1_L2",
               'S2': "COPERNICUS/S2_SR_HARMONIZED"},
        'TOA': {'L8': "LANDSAT/LC08/C02/T1_TOA",
                'L9': "LANDSAT/LC09/C02/T1_TOA",
                'S2': "COPERNICUS/S2_HARMONIZED"}}

    # Optical bands used to calculate MNDWI
    optical_bands = {
        'SR':{'L8': ['SR_B2','SR_B3', 'SR_B4','SR_B5', 
                       'SR_B6', 'QA_PIXEL'],
              'L9': ['SR_B2','SR_B3', 'SR_B4','SR_B5', 
                     'SR_B6', 'SR_QA_AEROSOL'],
              'S2': ['B2', 'B3', 'B4', 
                    'B8', 'B11']},
        'TOA':{'L8': ['B2','B3', 'B4','B5', 
                       'B6', 'B8', 'QA_PIXEL'],
               'L9': ['B2','B3', 'B4','B5', 
                      'B6', 'B8','QA_PIXEL'],
               'S2': ['B2', 'B3', 'B4', 
                      'B8', 'B8', 'B11']}
    }
    
    for mission in OPTICAL_MISSION:
        collection = ee.ImageCollection(
            collection_ids[PROCESS_STAGE][mission]).filterBounds(AOI).select(
                optical_bands[PROCESS_STAGE][mission]).filterDate(
                        START_DATE, END_DATE)
        if mission == OPTICAL_MISSION[0]:
            optical_collection = collection
        else:
            optical_collection = optical_collection.merge(collection)

    # Read optical image collection
    # optical_collection = ee.ImageCollection(
    #     collection_ids[PROCESS_STAGE][OPTICAL_MISSION]).filterBounds(AOI).select(
    #         optical_bands[PROCESS_STAGE][OPTICAL_MISSION]).filterDate(
    #                 START_DATE, END_DATE)
    
    # Add cloud and cloud mask bands
    if OPTICAL_MISSION[0] == 'S2':
        s2_cloudless_col = ee.ImageCollection(
            'COPERNICUS/S2_CLOUD_PROBABILITY').filterBounds(AOI).filterDate(
                    START_DATE, END_DATE)
        optical_collection = ee.ImageCollection(ee.Join.saveFirst(
            's2cloudless').apply(**{
                'primary': optical_collection,
                'secondary': s2_cloudless_col,
                'condition': ee.Filter.equals(**{
                    'leftField': 'system:index',
                    'rightField': 'system:index'
                })
            }))
        optical_collection = optical_collection.map(
            add_cloudMask_S2).map(scale_S2)
    else:
        optical_collection = optical_collection.map(
            add_cloudMask_Landsat)
        if PROCESS_STAGE == 'SR':
            optical_collection = optical_collection.map(scale_Landsat)
                        
    """
    ==========================
     Preprocess SAR images
    ==========================
    """
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
    
    crs = S1.first().projection().crs()
    
    """
    Define S1_ard parameters
    """   
    optical_dates = ee.List(
        optical_collection.aggregate_array('system:time_start')
        ).distinct()
    
        
    def date_filter(element):
        S2_filtered = optical_dates.filter(
            ee.Filter.rangeContains(
                'item',
                ee.Number(element).subtract(1000*3600*24),
                ee.Number(element)
                )
            )            

        date = ee.Algorithms.If(
            S2_filtered.size().lt(2),
            element,
            None)
        return date
    
    filter_dates = optical_dates.map(date_filter, True)
    print('{} of Optical images are found'.format(filter_dates.size().getInfo()))
    
    def pair_mosaic(element):
        # Filter and mosaic optical images on the same date
        date = ee.Date(element)
        optical_filtered = optical_collection.filterDate(
            date, 
            date.advance(1, 'day'))
        optical_bandnames = optical_filtered.first().bandNames()
        optical_composite = optical_filtered.reduce(
            ee.Reducer.first()).rename(optical_bandnames)

        if OPTICAL_MISSION[0] == 'S2':
            optical_id = optical_filtered.first().get('PRODUCT_ID')
        else:
            optical_id = optical_filtered.first().get('LANDSAT_PRODUCT_ID')
        
        img = optical_composite
        
        return ee.Image(img).copyProperties(
            optical_filtered.first()
            ).set({'system:time_start':element,
                   'opt_id':optical_id})
    
    optical_imgs = ee.ImageCollection(filter_dates.map(pair_mosaic)) 
    
    
    def cloud_cal(image):
        mask = image.select('Mask')
        reducer = ee.Reducer.sum().combine(ee.Reducer.count(), sharedInputs=True)
        cloud_stats = mask.reduceRegion(
            reducer,
            AOI,
            100,
            maxPixels=1e9)
        cloud_sum = ee.Number(cloud_stats.get('Mask_sum'))
        cloud_count = ee.Number(cloud_stats.get('Mask_count'))
        cloud_percent = cloud_sum.divide(cloud_count).multiply(100)
        return image.set({'CLOUD_PERCENTAGE_AOI': cloud_percent})         
    optical_imgs = optical_imgs.map(cloud_cal)                                        
    
    print('Retrieve metadata')
    optical_ids = optical_imgs.aggregate_array('opt_id').getInfo()
    cloud_cover = optical_imgs.aggregate_array('CLOUD_PERCENTAGE_AOI').getInfo()
    time = optical_imgs.aggregate_array('system:time_start').getInfo()
    meta_df = pd.DataFrame({
        'optical_id': optical_ids,
        'Cloud_percent': cloud_cover,
        'Time': time
        })
    meta_df.to_csv(os.path.join(OUT_DIR, 'optical.csv'), index=False)
    
    # Image download
    paths = {
        'Optical': os.path.join(OUT_DIR, '-'.join(OPTICAL_MISSION)),
        'Panchromatic': os.path.join(OUT_DIR, 'Panchromatic'),
        'Mask': os.path.join(OUT_DIR, 'Mask')
        }
    for key, path in paths.items():
        if not os.path.exists(path):
            os.mkdir(path)
    for index, row in meta_df.iterrows():
        filenames = {
            'Optical': row['optical_id'],
            'Panchromatic': row['optical_id']+'_Panchromatic',
            'Mask': row['optical_id']+'_Mask',       
            }
        bands = {
            'Optical': optical_bands[PROCESS_STAGE][OPTICAL_MISSION[0]][0:5],
            'Panchromatic': ['B8'],
            'Mask': ['QA_PIXEL','Mask']
            }
        
        for key in paths.keys():
            filepath = os.path.join(paths[key], filenames[key]+'.tif')
            if not os.path.exists(filepath):
                image = optical_imgs.filterMetadata(
                    'opt_id', 'equals', row['optical_id']).select(
                        bands[key]).first()
                if key == 'Mask':
                    image = image.toUint8()
                if (key == 'Optical')&(PROCESS_STAGE == 'TOA'):
                    image = image.toFloat()
                        
                done = False
                count = 0
                while not done:
                    try:
                        print('Donwload {}'.format(filenames[key]))
                        download_tif(image, AOI, crs,
                                     bands[key], filepath)
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