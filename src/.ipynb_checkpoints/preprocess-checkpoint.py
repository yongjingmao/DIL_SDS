# All the dataloaders are implemented in this file.
import sys
sys.path.append('../CoastSat')
from coastsat.SDS_shoreline import *

import argparse
from scipy.ndimage import gaussian_filter
import numpy as np
from skimage import transform as sk_transform
from skimage.io import imread
import os
import pandas as pd

import rasterio as rio
from rasterio.warp import reproject, Resampling


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, 
                        default='data/Narrabeen', 
                        help='Data path for optical images', metavar='')
    parser.add_argument('--cloud_ratio', type=float, 
                        default=0.5, 
                        help='Ratio of cloud to superimpose, rangeing between 0 and 1', metavar='')
    parser.add_argument('--temporal_var', action='store_true', 
                        help='Incoporate temporal variation')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def img_warp(img, profile, dst_profile, dst_fp):
    dst_shape = (profile['count'], dst_profile['height'], dst_profile['width'])
    destin = np.empty(dst_shape, dtype=profile['dtype'])

    reproject(
        img,
        destin,
        src_transform=profile['transform'],
        src_crs=profile['crs'],
        dst_transform=dst_profile['transform'],
        dst_crs=dst_profile['crs'],
        resampling=Resampling.cubic)

    profile['crs'] = dst_profile['crs']
    profile['transform'] = dst_profile['transform']
    profile['width'] = dst_profile['width']
    profile['height'] = dst_profile['height']

    with rio.open(dst_fp, 'w', **profile) as dst:
        dst.write(destin)
    return dst_fp

def output_transform(x):
    # normalize to max magnitude of 1
    x /= max([x.max(), -x.min()])
    # pass through tanh to ensure predefined range
    return np.tanh(4*x)

def interp(t):
    # --- Perlin Methods
    return 3 * t**2 - 2 * t ** 3

def perlin(width, height, scale=10, batch=1):
    """
    Generate perline noise
    """
    # based on https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869
    
    gx, gy = np.random.randn(2, batch, width + 1, height + 1, 1, 1)
    xs = np.linspace(0, 1, scale + 1)[:-1, None]
    ys = np.linspace(0, 1, scale + 1)[None, :-1]

    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    
    dots = 0
    dots += wx * wy * (gx[:,:-1, :-1] * xs + gy[:,:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[:,1:, :-1] * (1 - xs) + gy[:,1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:,:-1, 1:] * xs - gy[:,:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[:,1:, 1:] * (1 - xs) - gy[:,1:, 1:] * (1 - ys))

    return dots.transpose(0, 1, 3, 2, 4).reshape(batch, width * scale, height * scale)

def generate_perlin_noise_2d(scales=None, shape=(256,256), batch=1, weights=None, const_scale=True, decay_factor=1):
    
    """
    Generate 2D perline noise between [0, 1]
    """
    
    # Set Up Scales
    if scales is None:
        up_lim = max([2, int(np.log2(min(shape)))-1])        
        
        scales = [2**i for i in range(2,up_lim)]
        # proportional to image size, if const_scale is preserved
        if const_scale:
            f = int(2**np.floor(np.log2(0.25*max(shape)/max(scales))))
            scales = [el*f for el in scales]

    if weights is None:
        weights = [el**decay_factor for el in scales]
    # Round shape to nearest power of 2 
    big_shape = [int(2**(np.ceil(np.log2(i)))) for i in shape]
    out = np.zeros([batch,*shape])
    for scale, weight in zip(scales, weights):
        out += weight*perlin(int(big_shape[0]/scale), int(big_shape[1]/scale), scale=scale, batch=batch)[...,:shape[0],:shape[1]]

    return (output_transform(out)[0, :, :] + 1)/2



def add_cloud(clear_img, cloud_imgs, ratio=0.5):
    """
    Add Clouds
    """
    # Making permanent masks, same dize as that images i.e. 300x300
    # these masks are then applied for alpha blending with cloud images. The masks are later resized to the size
    # of cloud
    if ratio == 0:
        mask = np.zeros((clear_img.shape[1], clear_img.shape[2]))
        alpha = np.zeros((clear_img.shape[1], clear_img.shape[2]))
    elif ratio == 1:
        mask = np.ones((clear_img.shape[1], clear_img.shape[2]))
        alpha = np.ones((clear_img.shape[1], clear_img.shape[2]))
    else:   
        threshold = 1 - ratio
        perlin_noise = generate_perlin_noise_2d(shape=(clear_img.shape[1], clear_img.shape[2]))
        mask = perlin_noise>threshold


        perlin_noise = np.clip(perlin_noise, threshold, 1)
        alpha = np.interp(perlin_noise, (perlin_noise.min(), perlin_noise.max()), (0, 1))


    # Randomly pick a cloud image
    percentile = np.random.randint(80, 100)
    cloud_img = np.percentile(cloud_imgs, percentile, axis=(0))


    # Alpha blending
    blend_img = (alpha) * cloud_img + (1 - alpha) * clear_img
            
    return blend_img, mask

def rescale(array, target_mean, target_std):
    # Calculate the min and max of the column
    current_mean = array.mean()
    current_std = array.std()

    # Handle edge case where std is zero
    if current_std == 0:
        # If current std is zero, return a column where all values are target_mean
        return pd.Series([target_mean] * len(col_values), index=df.index)

    # Standardize the column to mean 0 and std 1
    standardized = (array - current_mean) / current_std

    # Scale to the new mean and std
    scaled = standardized * target_std + target_mean

    return scaled

def main(args):
    data_path = args.data_path
    cloud_ratio = args.cloud_ratio
    temporal_var = args.temporal_var
    
    in_opt_folder = os.path.join(data_path, 'S1_Landsat', 'Optical')
    in_sar_folder = os.path.join(data_path, 'S1_Landsat',' SAR')
    in_mask_folder = os.path.join(data_path, 'S1_Landsat', 'Mask')

    if temporal_var:
        out_suffix = 'var'
    else:
        out_suffix = ''

    out_opt_folder  = os.path.join(data_path, 'S1_Landsat', "Optical_{:d}{}".format(int(cloud_ratio*100), out_suffix))
    out_mask_folder = os.path.join(data_path, 'S1_Landsat', "Mask_{:d}{}".format(int(cloud_ratio*100), out_suffix))
    out_mndwi_folder = os.path.join(data_path, 'S1_Landsat', "MNDWI_{:d}{}".format(int(cloud_ratio*100), out_suffix))

    if not os.path.exists(out_opt_folder):
        os.mkdir(out_opt_folder)
        
    if not os.path.exists(out_mask_folder):
        os.mkdir(out_mask_folder)
    
    if not os.path.exists(out_mndwi_folder):
        os.mkdir(out_mndwi_folder)


    # Read metadata
    opt_df = pd.read_csv(os.path.join(data_path, 'S1_Landsat', 'optical.csv'))
    S1_df = pd.read_csv(os.path.join(data_path, 'S1_Landsat', 'S1.csv'))

    folders = {
        'optical_id': in_opt_folder,
        'SAR_id': in_sar_folder,
        'Mask_id': in_mask_folder
    }


    """
    Pair SAR and optical images and save metadata
    """
    pair_fp = os.path.join(data_path, 'S1_Landsat', 'pair.csv')
    pair_window = 24*3600*1000 # Convert 1 day time window to millisecond 
    
    paired_S1 = []
    for time in opt_df['Time']:
        pair_filter = (S1_df['Time']>time-pair_window)&(S1_df['Time']<time+pair_window)
        if sum(pair_filter)>0:
            paired_S1.append(S1_df[pair_filter]['SAR_id'].to_numpy()[0])
        else:
            paired_S1.append('')
    
    opt_df['SAR_id'] = paired_S1
    opt_df['Time'] = pd.to_datetime(opt_df['Time'], unit='ms')
    pair_df = opt_df.sort_values('Time')    
    pair_df = pair_df[pair_df['SAR_id']!=''].reset_index(drop=True)
    pair_df.to_csv(pair_fp, index=False)

    """
    Warp optical images to SAR image
    """
    print('Warp optical images to SAR image')
    target_fp = os.path.join(in_sar_folder, pair_df.loc[0, 'SAR_id']+'.tif')
    with rio.open(target_fp) as src_dst:
        dst_img = src_dst.read()
        dst_profile  = src_dst.meta

    for i, row in opt_df.iterrows():
        for column in ['optical_id', 'Mask_id']:
            if column == 'Mask_id':
                source_fp = os.path.join(folders[column], row['optical_id']+'_Mask.tif')
            else:
                source_fp = os.path.join(folders[column], row[column]+'.tif')
                
            with rio.open(source_fp) as src_source:
                profile = src_source.meta
                img = src_source.read()
                
            if not (profile['width']==dst_profile['width'])&(profile['height'] == dst_profile['height'])&(profile['crs'] == dst_profile['crs']):
                dst_fp = img_warp(img, profile, dst_profile, source_fp)

    
    """
    Add synthetic clouds and calculate MNDWI
    """
    print('Add synthetic clouds to clear optical images')
    cloudfree_df = opt_df.loc[opt_df['Cloud_percent']<20, :]
    cloudfree_name = list(cloudfree_df['optical_id']+'.tif')
    
    cloudfull_df = opt_df.loc[opt_df['Cloud_percent']>80, :]
    cloudfull_name = list(cloudfull_df['optical_id']+'.tif')
    
    if temporal_var:
        opt_df['Datetime'] = pd.to_datetime(opt_df['Time'], unit='ms')
        opt_df['Season'] = opt_df['Datetime'].dt.month%12 // 3 + 1
        seasonal_cloud = opt_df[['Season', 'Cloud_percent']].groupby('Season').median()
        seasonal_cloud['Cloud_scale'] = rescale(seasonal_cloud['Cloud_percent'], cloud_ratio, 0.2)
        opt_df = pd.merge(opt_df, seasonal_cloud[['Cloud_scale']], on='Season', how='left')
    else:
        opt_df['Cloud_scale'] = cloud_ratio

    cloud_imgs = []
    for name in cloudfull_name:
        in_opt_file = os.path.join(in_opt_folder, name)
        with rio.open(in_opt_file) as src:
            cloud_imgs.append(src.read())
    cloud_imgs = np.stack(cloud_imgs)

    for i, row in opt_df.iterrows():
        basename = row['optical_id'] + '.tif'
        maskname = basename.replace('.tif', '_Mask.tif')
        mndwiname = basename.replace('.tif', '_MNDWI.tif')
        in_opt_file = os.path.join(in_opt_folder, basename)
        in_mask_file = os.path.join(in_mask_folder, maskname)
        out_opt_file = os.path.join(out_opt_folder, basename)
        out_mask_file = os.path.join(out_mask_folder, maskname)
        out_mndwi_file = os.path.join(out_mndwi_folder, mndwiname)
        
        with rio.open(in_mask_file) as src:
            mask = src.read([2]).astype('uint8')
        
        with rio.open(in_opt_file) as src:
            profile = src.meta
            img = src.read().astype(float)
            if basename in cloudfree_name:
                new_img, mask_simu = add_cloud(img, cloud_imgs, row['Cloud_scale'])
                mask_simu = np.expand_dims(mask_simu, 0)
                mask = (mask + mask_simu)>=1
                mask = mask.astype('uint8')
    
            else:
                new_img = img
                
            mask_profile = profile.copy()
            mask_profile.update({'count':1, 'dtype':'uint8', 'nodata': 2})
            
            mndwi_profile = mask_profile.copy()
            mndwi_profile['dtype'] = 'float32'
    
            with rio.open(out_opt_file, 'w', **profile) as dst:
                dst.write(new_img)
    
            with rio.open(out_mask_file, 'w', **mask_profile) as dst:
                dst.write(mask)
    
            im = new_img.transpose(1,2,0)
            im_mndwi = SDS_tools.nd_index(im[:,:,4], im[:,:,1], mask[0]==1)
            im_mndwi = np.where(np.isnan(im_mndwi), -1, im_mndwi).astype(np.float32)
            im_mndwi = np.expand_dims(im_mndwi, 0)            
            with rio.open(out_mndwi_file, 'w', **mndwi_profile) as dst:
                dst.write(im_mndwi)


if __name__ == '__main__':
    args = parse_args()
    main(args)