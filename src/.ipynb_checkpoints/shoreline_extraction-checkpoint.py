import sys
import os
import glob
import argparse

import rasterio
from rasterio.coords import BoundingBox
from rasterio.features import rasterize
import cv2
import geopandas as gpd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append('../CoastSat')
from coastsat import SDS_transects
from coastsat.SDS_shoreline import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, 
                        default='data/Narrabeen/S1_Landsat', 
                        help='Data path for cloudy images', metavar='')
    parser.add_argument('--result_path', type=str, 
                        default='results/Narrabeen', 
                        help='Data path for model predictions', metavar='')
    parser.add_argument('--cloud_ratio', type=float, 
                        default=0.5, 
                        help='Ratio of cloud to superimpose, rangeing between 0 and 1', metavar='')
    parser.add_argument('--num_pass', type=int, 
                        default=10, 
                        help='Number of passes in DIL run', metavar='')
    parser.add_argument('--max_dist_ref', type=int, 
                        default=100, 
                        help='Maximum allowed distance [m] to reference shoreline', metavar='')
    parser.add_argument('--min_length', type=int, 
                        default=50, 
                        help='Minimum allowed shoreline length [m]', metavar='')

    

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def retrieve_SDS(img, mask, crs, georef, min_length=100, ref_shoreline=None, max_dist_ref=150):
    settings = {
        'output_epsg':crs,
        'min_length_sl':min_length,
        'dist_clouds': 30,
    }
    if ref_shoreline is not None:
        settings['reference_shoreline'] = ref_shoreline
        settings['max_dist_ref'] = max_dist_ref

    im_ref_buffer = create_shoreline_buffer(mask.shape, georef, crs, 10, settings)
    if np.sum(mask == 0) > 5000:  # Clear pixels above 5000
        contours_mwi, t_mndwi = find_wl_contours1(img, mask, im_ref_buffer)
        shoreline = process_shoreline(contours_mwi, mask, np.zeros(mask.shape), georef, crs, settings)
    else:  # Too few clear pixels to derive shoreline position
        shoreline = np.array([[],[]]).transpose()
    return shoreline


def cal_position(shorelines, transects, time):
    settings_transects = {  # parameters for computing intersections
                      'along_dist':          25,        # along-shore distance to use for computing the intersection
                      'min_points':          3,         # minimum number of shoreline points to calculate an intersection
                      'max_std':             15,        # max std for points around transect
                      'max_range':           30,        # max range for points around transect
                      'min_chainage':        -100,      # largest negative value along transect (landwards of transect origin)
                      'multiple_inter':      'auto',    # mode for removing outliers ('auto', 'nan', 'max')
                      'auto_prc':            0.1,       # percentage of the time that multiple intersects are present to use the max
                     }
    model_shoreline_dict = {'shorelines': shorelines}
    model_distance = SDS_transects.compute_intersection_QC(
        model_shoreline_dict, transects, settings_transects)
    model_result = {'Time': time}
    model_result.update(model_distance)
    model_result = pd.DataFrame(model_result)
    return model_result


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

def nan_diff(series):
    # Drop NaN and compute diff
    diff_series = series.dropna().diff()
    # Reindex to the original series
    return diff_series.reindex(series.index)
    
    
def main(args):

    """
    Read data
    """
    data_path = args.data_path
    result_path = args.result_path
    cloud_ratio = args.cloud_ratio
    num_pass = args.num_pass
    max_dist_ref = args.max_dist_ref
    min_length = args.min_length
    
    opt_folder = os.path.join(data_path, 'S1_Landsat', 'Optical')
    mask_folder = os.path.join(data_path, 'S1_Landsat', 'Mask')
    synmask_folder = os.path.join(data_path, 'S1_Landsat', "Mask_{:d}".format(int(cloud_ratio*100)))
    
    # Read metadata
    opt_df = pd.read_csv(os.path.join(data_path, 'S1_Landsat', 'optical.csv'))
    opt_df['Time'] = pd.to_datetime(opt_df['Time'], unit='ms')
    pred_ids = opt_df['Time'].dt.strftime('%Y-%m-%d')
    opt_df['pred_id'] = pred_ids

    # Read an image for reference
    with rasterio.open(glob.glob(os.path.join(result_path, '**', '*.tif'))[0]) as src:
        crs = src.crs
        resize = (src.meta['height'] , src.meta['width'])

    # Read transects and reference shorelines
    gdf_transects = gpd.read_file(os.path.join(data_path, 'transects.geojson')).to_crs(crs)
    transects = dict([])
    for i in gdf_transects.index:
        transects[gdf_transects.loc[i,'TransectId']] = np.array(gdf_transects.loc[i,'geometry'].coords)
        
    gdf_ref = gpd.read_file(os.path.join(data_path, 'ref_shoreline.geojson')).to_crs(crs)
    ref_line_coords = np.array(gdf_ref.loc[0,'geometry'].geoms[0].coords)

    # Create buffer around ref line
    buffer = gdf_ref.buffer(200)
    buffer_raster = polygon_to_binary_image(buffer[0], src)
    buffer_mask, _ = crop_and_resize(buffer_raster, resize, src.meta)



    """
    # Retrieve shoreline for model results
    """
    
    print('Extract shorelines for model results')
    for pass_num in range(1, num_pass+1):
        pred_folder = os.path.join(result_path, '{:03d}'.format(pass_num))
        if not os.path.exists(pred_folder+'/SDS_clear.csv'):
            shorelines_clear = []
            shorelines_cloud = []
            times = []
            for i, row in opt_df.iterrows():
                img_name = row['pred_id'] + '.tif'
                mask_name = row['optical_id'] + '_Mask.tif'

                if os.path.exists(os.path.join(pred_folder, img_name)):
                    # Read and model result
                    with rasterio.open(os.path.join(pred_folder, img_name)) as src:
                        img = src.read([1])
                        affine = np.array(list(src.transform))
                        georef = list(affine[[2, 0, 1, 5, 3, 4]])
                        img = np.where(buffer_mask==1, img, np.nan)
    
                    # Read synthetic cloud mask
                    with rasterio.open(os.path.join(synmask_folder, mask_name)) as src:
                        mask = src.read([1]).astype(np.float32)
                        mask, _ = crop_and_resize(mask, resize, src.meta)
                        mask = (mask>=0.5).astype(np.float32)
                        mask_clear = np.where(buffer_mask==1, mask[0] == 0, True)[0]  # Clear pixels masked
                        mask_cloud = np.where(buffer_mask==1, mask[0] == 1, True)[0]  # Cloud pixels masked
                        
        
                    shoreline_clear = retrieve_SDS(img[0], mask_cloud, crs=int(src.crs.to_string()[5:]), 
                                                   georef=georef, min_length=min_length,
                                                   ref_shoreline=ref_line_coords, max_dist_ref=max_dist_ref)
                    shoreline_cloud = retrieve_SDS(img[0], mask_clear, crs=int(src.crs.to_string()[5:]), 
                                                   georef=georef, min_length=min_length, 
                                                   ref_shoreline=ref_line_coords, max_dist_ref=max_dist_ref)
                    
                    shorelines_clear.append(shoreline_clear)
                    shorelines_cloud.append(shoreline_cloud)
                    times.append(row['Time'])

            result_clear = cal_position(shorelines_clear, transects, times)
            result_cloud = cal_position(shorelines_cloud, transects, times)
            result_clear.to_csv(pred_folder+'/SDS_clear.csv', index=False)
            result_cloud.to_csv(pred_folder+'/SDS_cloud.csv', index=False)

    """
    # Retrieve shoreline for obs results
    """
    
    print('Extract shorelines for target results')
    if not os.path.exists(opt_folder+'/SDS.csv'):
        shorelines_true = []
        times = []
        for i, row in opt_df.iterrows():
            img_name = row['optical_id'] + '.tif'
            if os.path.exists(os.path.join(opt_folder, img_name)):
                with rasterio.open(os.path.join(opt_folder, img_name)) as src:
                    img = src.read().transpose(1,2,0)
                    affine = np.array(list(src.transform))
                    georef = list(affine[[2, 0, 1, 5, 3, 4]])
                with rasterio.open(os.path.join(mask_folder, img_name.replace('.tif', '_Mask.tif'))) as src:
                    mask = src.read(2) == 1
                    mask = np.where(buffer_raster==1, mask, True)[0]
        
                im_mndwi = SDS_tools.nd_index(img[:,:,4], img[:,:,1], mask)
                im_mndwi = np.where(buffer_raster==1, im_mndwi, np.nan)[0]
                shoreline = retrieve_SDS(im_mndwi, mask, crs=int(src.crs.to_string()[5:]), 
                                         georef=georef, min_length=min_length, ref_shoreline=ref_line_coords,
                                         max_dist_ref=max_dist_ref)
                shorelines_true.append(shoreline)
    
        result_true = cal_position(shorelines_true, transects, times)
        diff_true = result_true.apply(nan_diff)
        result_true = result_true.mask(diff_true.abs()>100)
        result_true.to_csv(opt_folder+'/SDS.csv')

    """
    # Find best SDS results from all passes
    """
    result_true = pd.read_csv(opt_folder+'/SDS.csv', parse_dates=['Time']
                             ).set_index('Time').sort_index()
    for pass_num in range(1, num_pass+1):
        pred_folder = os.path.join(result_path, '{:03d}'.format(pass_num))
        result_clear = pd.read_csv(pred_folder+'/SDS_clear.csv', parse_dates=['Time']
                                  ).set_index('Time').sort_index()
        result_cloud = pd.read_csv(pred_folder+'/SDS_cloud.csv', parse_dates=['Time']
                          ).set_index('Time').sort_index()

        if len(model_result)>1:
            diff = (result_true - result_cloud).values.reshape(-1)
            diff = diff[~np.isnan(diff)]
            mae = np.mean(abs(diff))
            if mae < best_MAE:
                best_pass = pass_num
                best_MAE = mae
                result_clear.to_csv(os.path.join(result_path, 'SDS_clear.csv'))
                result_cloud.to_csv(os.path.join(result_path, 'SDS_cloud.csv'))

    """
    # Plot results
    """
    result_true = pd.read_csv(clear_folder+'/SDS.csv').set_index('Time').sort_index()
    result_clear = pd.read_csv(result_path, 'SDS_clear.csv').set_index('Time').sort_index()
    result_cloud = pd.read_csv(result_path, 'SDS_cloud.csv').set_index('Time').sort_index()
    
    diff_cloud = (result_cloud - result_true).values.reshape(-1)
    diff_cloud = diff_cloud[~np.isnan(diff_cloud)]
    
    diff_clear = (result_clear - result_true).values.reshape(-1)
    diff_clear = diff_clear[~np.isnan(diff_clear)]

    # Plot histograms
    fig = plt.figure(figsize=[10, 3], tight_layout=True)
    colours = ['r', 'g']
    gs = gridspec.GridSpec(1,2)
    gs.update(left=0.04, right=0.97, bottom=0.06, top=0.95, hspace=0.4)
    
    diffs = {'Under Synthetic Clouds': diff_cloud,
            'Outside Synthetic Clouds': diff_clear}
    
    for k, key in enumerate(diffs.keys()):
        
        diff = diffs[key]
        # get metrics
        mae = np.mean(abs(diff))
        mean = np.mean(diff)
        std = np.std(diff)
        
        # plot histogram
        ax = fig.add_subplot(gs[0,k])
        ax.axvline(x=0, ls='--', lw=1.5, color='k')
        binwidth = 1
        bins = np.arange(min(diff), max(diff) + binwidth, binwidth)
        density = ax.hist(diff, bins=bins, density=True, color=colours[k],
                          edgecolor='k', alpha=0.5, label=key)
        mu, std = stats.norm.fit(diff)
        pval = stats.normaltest(diff)[1]
        x = np.linspace(-20, 20, 100)
        p = stats.norm.pdf(x, mu, std)
        ax.plot(x, p, '-',color=colours[k], linewidth=1)
        ax.set(xlabel='Shoreline position error (m)', xlim=[-20, 20])
        str_stats = ' MAE = %.1f\n Mean = %.1f\n STD = %.1f\n n = %d' % (mae, mean, std, len(diff))
        ax.text(0.02, 0.94, str_stats,va='top', transform=ax.transAxes,fontsize=11,
                bbox=dict(boxstyle="square", ec='k',fc='w',alpha=0.75))
        ax.legend(loc='upper right',edgecolor='k')
        ax.set_yticks([])

    fig.savefig(os.path.join(result_path, 'error_hists.jpg'),
                dpi=300, bbox_inches='tight')

    # Plot time series
    fig, axes = plt.subplots(3, 2, figsize= (14, 8))
    tran_ids = list(gdf_transects.iloc[np.linspace(1, len(transects)-2, 5).astype(int),
                    np.where(gdf_transects.columns=='TransectId')[0][0]])
    time = pd.to_datetime(result_true.index)

    for i, tran_id in enumerate(tran_ids):
        ax = axes[i//2, i%2] 
        nan_filter = ~result_true[tran_id].isna()
        ax.plot(time[nan_filter], 
                result_true[tran_id][nan_filter], color='k', marker='s', 
                label='Target shorelines', zorder=-1)

        ax.scatter(time[(nan_filter)], 
                   result_cloud[tran_id][nan_filter.values], s=20,
                   color='r', label='Modelled shorelines under Synthetic clouds')
        ax.scatter(time[(nan_filter)],
                   result_clear[tran_id][nan_filter.values], s=20,
                   color='g', label='Modelled shorelines outside Synthetic clouds')
        MAE_cloud = (result_cloud[tran_id][nan_filter.values] - result_true[tran_id][nan_filter]).abs().mean()
        MAE_clear = (result_clear[tran_id][nan_filter.values] - result_true[tran_id][nan_filter]).abs().mean()

        title_y = 1.05  # Adjust this to control the title's y position
        title_x = 0.5  # Centered by default

        # Set the common part of the title
        ax.text(title_x-0.25, title_y, tran_id+': ', transform=ax.transAxes, fontsize=10, ha='center')
        # Add the first part of the colored text
        ax.text(title_x-0.07, title_y, 'MAE (under)=%.1f,' % MAE_cloud, color='red', 
                transform=ax.transAxes, fontsize=10, ha='center')
        # Add the second part of the colored text
        ax.text(title_x+0.2, title_y, ' MAE (outside)=%.1f' % MAE_clear, color='green', 
                transform=ax.transAxes, fontsize=10, ha='center')
        ax.set_ylabel('Shoreline position (m)')
        ax.set_ylim(160, 265)

        if i==4:
            ax.legend(ncol=1, loc=1, bbox_to_anchor=(2, 1))
        if i%2==1:
            ax.set_ylabel('')

    axes[2, 1].axis("off")
    plt.subplots_adjust(hspace=0.3)
    fig.savefig(os.path.join(result_path, 'timeseries.jpg'),
                dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    args = parse_args()
    main(args)