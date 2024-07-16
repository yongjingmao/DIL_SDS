import torch
import numpy as np
import os
from scipy import ndimage
import cv2
import rasterio
from rasterio.coords import BoundingBox
from rasterio.features import rasterize
import random
import pandas as pd
import geopandas as gpd

from flow_estimator import FlowEstimator
from models.perceptual import LossNetwork
from utils import *

class Dataset(object):
    """
    Data loader for the input video 
    """    
    def __init__(self, cfg):
        self.cfg = cfg
        if not os.path.exists(self.cfg['data_path']):
            raise Exception("Input data path not found: {}".format(self.cfg['data_path']))
        
        # Read sar and optical metadata
        self.sar_meta_fp = os.path.join(self.cfg['data_path'], 'S1_Landsat', 'S1.csv')
        self.optical_meta_fp = os.path.join(self.cfg['data_path'], 'S1_Landsat', 'optical.csv')
        self.ref_line_fp = os.path.join(self.cfg['data_path'], 'ref_shoreline.geojson')

        S1_df = pd.read_csv(self.sar_meta_fp)
        opt_df = pd.read_csv(self.optical_meta_fp)
        pair_window = 24*3600*1000 # Only Pair S1 amd optical images collected on the same day
        
        paired_S1 = []
        for time in opt_df['Time']:
            pair_filter = (S1_df['Time']>time-pair_window)&(S1_df['Time']<time+pair_window)
            if sum(pair_filter)>0:
                paired_S1.append(S1_df[pair_filter]['SAR_id'].to_numpy()[0])
            else:
                paired_S1.append('')

        opt_df['SAR_id'] = paired_S1
        opt_df['Time'] = pd.to_datetime(opt_df['Time'], unit='ms')
        meta_df = opt_df.sort_values('Time')
        
        if self.cfg['paired']:
            # Only use paired images
            self.meta = meta_df[meta_df['SAR_id']!=''].reset_index(drop=True)
        else:
            self.meta = meta_df.reset_index(drop=True)
        
        self.sar_path = os.path.join(self.cfg['data_path'], 'S1_Landsat', 'SAR')
        self.opt_path = os.path.join(self.cfg['data_path'], 'S1_Landsat', 'MNDWI_{}'.format(int(self.cfg['cloud_ratio']*100)))
        self.mask_path = os.path.join(self.cfg['data_path'], 'S1_Landsat', 'Mask_{}'.format(int(self.cfg['cloud_ratio']*100)))
        

        
        with rasterio.open(os.path.join(self.opt_path, self.meta['optical_id'][0]+'_MNDWI.tif')) as src:
            self.frame_W = src.width
            self.frame_H = src.height
        

        if not self.cfg['resize'] is None:
            self.frame_H, self.frame_W = self.cfg['resize']
        self.frame_sum =  len(self.meta)
        self.frame_size = self.cfg['frame_size'] = (self.frame_H , self.frame_W)
        self.batch_size = self.cfg['batch_size']
        self.batch_idx = 0
        self.batch_list_train = None

        if self.cfg['use_perceptual']:
            self.netL = LossNetwork(None, self.cfg['perceptual_layers'])
        
        if (self.cfg['input_type'] == 'Noise')|(~self.cfg['paired']):
            self.init_noise()

            
        self.init_data()
        self.init_flow()
        self.init_flow_mask()
        self.init_perceptual_mask()
        self.init_batch_list()


    def init_data(self):
        """
        Load input video and mask
        """
        self.image_all = [] # S2 images
        self.mask_all = [] # S2 Mask
        self.input_all = [] # S1 images
        self.contour_all = [] # Edge of Mask
        self.profile_all = [] # Image profile
        self.imagename_all = [] # Image names
        
        for i, row in self.meta.iterrows():
            opt_id = row['optical_id']+'_MNDWI.tif'
            mask_id = row['optical_id']+'_Mask.tif'
            time = row['Time']
            self.imagename_all.append(time.strftime('%Y-%m-%d')) 
            
            # Read S2 data as target
            frame, profile = self.init_optical(opt_id)                                                              
            self.profile_all.append(profile)
            self.image_all.append(frame)
            
            # Read mask data
            mask, contour = self.init_mask(mask_id, ref_line_path=self.ref_line_fp)
            self.mask_all.append(mask)
            self.contour_all.append(contour)
                        
            
            if self.cfg['paired']:
                # If only paired images were used
                # Use SAR or noise data as input
                if self.cfg['input_type'] == 'Noise':
                    self.input_all.append(self.input_noise[i])
                else:
                    sar_id = row['SAR_id']+'.tif'
                    self.input_all.append(self.init_sar(sar_id))
            else:
                # If both paired and unpaired images were used
                # Use either SAR or noise according to whether
                # there is a SAR paired
                if self.cfg['input_type'] == 'Noise':
                    self.input_all.append(self.input_noise[i])
                else:
                    if row['SAR_id']=='':
                        self.input_all.append(self.input_noise[i])
                    else:
                        sar_id = row['SAR_id']+'.tif'
                        self.input_all.append(self.init_sar(sar_id))
                
         
        self.image_all = np.array(self.image_all)
        self.mask_all = np.array(self.mask_all)
        self.input_all = np.array(self.input_all)
        
        self.profile = profile
        
    
    def init_optical(self, opt_id):
        """
        Read and normalize optical data
        """
        with rasterio.open(os.path.join(self.opt_path, opt_id)) as src:
            frame = src.read([1])
            profile = src.meta.copy()
            frame, transform = self.crop_and_resize(frame, self.frame_size, profile)

            profile['count'] = 1
            profile['dtype'] = 'float32'
            profile['width'] = self.frame_W
            profile['height'] = self.frame_H
            profile['transform'] = transform

            # Clip and normalize images
            frame = (frame + 1)/2
            frame = np.clip(frame, 0, 1)
        
        return frame, profile
    
    def init_mask(self, mask_id, ref_line_path=None):
        """
        Read mask data
        """
        with rasterio.open(os.path.join(self.mask_path, mask_id)) as src:
            mask = src.read([1]).astype(np.float32)
            if ref_line_path is not None:
                ref_line = gpd.read_file(ref_line_path).to_crs(src.crs)
                buffer = ref_line.geometry[0].buffer(200)
                buffer_raster = self.polygon_to_binary_image(buffer, src)
                mask = mask + (1-buffer_raster)
                mask[mask>1] = 1
            
            mask, _ = self.crop_and_resize(mask, self.frame_size, src.meta, interpolation=cv2.INTER_NEAREST)
            #mask = (mask>=0.5).astype(np.float32)
            
            if self.cfg['dilation_iter'] > 0:
                mask = ndimage.binary_dilation(mask==1, iterations=self.cfg['dilation_iter']).astype(np.float32)
            contour, hier = cv2.findContours(mask[0].astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return mask, contour
        

    def init_noise(self):
        """
        Generate input noise map
        """
        input_noise = get_noise(self.frame_sum, self.cfg['input_channel'], 'noise', self.frame_size, var=self.cfg['input_ratio']).float().detach()
        input_noise = torch_to_np(input_noise) # N x C x H x W
        self.input_noise = input_noise / self.cfg['input_ratio']
        
    
    def init_sar(self, sar_id):
        """
        Read and normalize SAR data
        """           
        with rasterio.open(os.path.join(self.sar_path, sar_id)) as src:
            input_sar = src.read([1, 2]).astype(np.float32)
            VV = input_sar[0].copy()
            VH = input_sar[1].copy()


            VV = np.clip(VV, -25.0, 0)
            VV = (VV + 25.1)/ 25.1
            VH = np.clip(VH, -32.5, 0)
            VH = (VH + 32.6)/32.6

            input_sar[0] = VV
            input_sar[1] = VH
            input_sar, _ = self.crop_and_resize(input_sar, self.frame_size, src.meta)
                
        return input_sar


    def init_flow(self):
        """
        Estimate flow using PWC-Net
        """
        self.cfg['flow_value_max'] = self.flow_value_max = None
        if self.cfg['use_flow']:
            flow_estimator = FlowEstimator()
            print('Loading input video and estimating flow...')
            self.flow_all = { ft : [] for ft in self.cfg['flow_type']}
            self.flow_value_max = {}

            for bs in self.cfg['batch_stride']:
                f, b = 'f' + str(bs), 'b' + str(bs)
                for fid in range(0, self.frame_sum - bs):
                        frame_first = np_to_torch(self.image_all[fid]).clone()
                        frame_second = np_to_torch(self.image_all[fid + bs]).clone()
                        frame_first = frame_first.cuda()
                        frame_second = frame_second.cuda()
                        flow_f = flow_estimator.estimate_flow_pair(frame_first, frame_second).detach().cpu()
                        flow_b = flow_estimator.estimate_flow_pair(frame_second, frame_first).detach().cpu()
                        torch.cuda.empty_cache()

                        flow_f, flow_b = check_flow_occlusion(flow_f, flow_b)
                        self.flow_all[f].append(flow_f.numpy())
                        self.flow_all[b].append(flow_b.numpy())

            for bs in self.cfg['batch_stride']:
                f, b = 'f' + str(bs), 'b' + str(bs)
                self.flow_all[f] = np.array(self.flow_all[f] + [self.flow_all[f][0]] * bs, dtype=np.float32)
                self.flow_all[b] = np.array([self.flow_all[b][0]] * bs + self.flow_all[b], dtype=np.float32)
                self.flow_value_max[bs] = max(np.abs(self.flow_all[f]).max().astype('float'), \
                    np.abs(self.flow_all[b]).max().astype('float'))
            self.cfg['flow_value_max'] = self.flow_value_max


    def init_flow_mask(self):
        """
        Pre-compute warped mask and intersection of warped mask with original mask
        """
        if self.cfg['use_flow']:
            self.mask_warp_all = { ft : [] for ft in self.cfg['flow_type']}
            self.mask_flow_all = { ft : [] for ft in self.cfg['flow_type']}
            for bs in self.cfg['batch_stride']:
                    f, b = 'f' + str(bs), 'b' + str(bs)
                    # forward
                    mask_warpf, _ = warp_np(self.mask_all[bs:], self.flow_all[f][:-bs][:, :2, ...])
                    mask_warpf = (mask_warpf > 0).astype(np.float32)
                    mask_flowf = 1. - (1. - mask_warpf) * (1. - self.mask_all[:-bs])
                    self.mask_warp_all[f] = np.concatenate((mask_warpf, self.mask_all[-bs:]), 0)
                    self.mask_flow_all[f] = np.concatenate((mask_flowf, self.mask_all[-bs:]), 0)
                    # backward
                    mask_warpb, _ = warp_np(self.mask_all[:-bs], self.flow_all[b][bs:][:, :2, ...])
                    mask_warpb = (mask_warpb > 0).astype(np.float32)
                    mask_flowb = 1. - (1. - mask_warpb) * (1. - self.mask_all[bs:])
                    self.mask_warp_all[b] = np.concatenate((self.mask_all[:bs], mask_warpb), 0)
                    self.mask_flow_all[b] = np.concatenate((self.mask_all[:bs], mask_flowb), 0)


    def init_perceptual_mask(self):
        """
        Pre-compute shrinked mask for perceptual loss
        """
        if self.cfg['use_perceptual']:
            self.mask_per_all = []
            mask_per = self.netL(np_to_torch(self.mask_all))
            for i, mask in enumerate(mask_per):
                self.mask_per_all.append((mask.detach().numpy() > 0).astype(np.float32))

    
    def init_batch_list(self):
        """
        List all the possible batch permutations
        """
        if self.cfg['use_flow']:
            self.batch_list = []
            for flow_type in self.cfg['flow_type']:
                batch_stride = int(flow_type[1])
                for batch_idx in range(0, self.frame_sum - (self.batch_size - 1) * batch_stride, self.cfg['traverse_step']):
                    self.batch_list.append((batch_idx, batch_stride, [flow_type]))
            if self.cfg['batch_mode'] == 'random':
                random.shuffle(self.batch_list)
        else:
            for bs in self.cfg['batch_stride']:
                self.batch_list = self.batch_list = [(i, bs, []) for i in range(self.frame_sum - self.batch_size + 1)]
            if self.cfg['batch_mode'] == 'random':
                median = self.batch_list[len(self.batch_list) // 2]
                random.shuffle(self.batch_list)
                self.batch_list.remove(median)
                self.batch_list.append(median)


    def set_mode(self, mode):
        if mode == 'infer':
            self.batch_list_train = self.batch_list
            self.batch_list = [(i, 1, ['f1']) for i in range(self.frame_sum - self.batch_size + 1)]
        elif mode == 'train':
            if not self.batch_list_train is None:
                self.batch_list = self.batch_list_train
            else:
                self.init_batch_list()


    def next_batch(self):
        if len(self.batch_list) == 0:
            self.init_batch_list()
            return None
        else:
            (batch_idx, batch_stride, flow_type) = self.batch_list[0]
            self.batch_list = self.batch_list[1:]
            return self.get_batch_data(batch_idx, batch_stride, flow_type)


    def get_batch_data(self, batch_idx=0, batch_stride=1, flow_type=[]):
        """
        Collect batch data for centain batch 
        """
        cur_batch = range(batch_idx, batch_idx + self.batch_size*batch_stride, batch_stride)
        batch_data = {}
        input_batch, img_batch, mask_batch, contour_batch = [], [], [], []
        profile_batch, imgname_batch = [], []
        if self.cfg['use_flow']:
            flow_batch = { ft : [] for ft in flow_type}
            mask_flow_batch = { ft : [] for ft in flow_type}
            mask_warp_batch = { ft : [] for ft in flow_type}
        if self.cfg['use_perceptual']:
            mask_per_batch = [ [] for _ in self.cfg['perceptual_layers']]

        for i, fid in enumerate(cur_batch):
            input_batch.append(self.input_all[fid])
            img_batch.append(self.image_all[fid])    
            mask_batch.append(self.mask_all[fid])            
            contour_batch.append(self.contour_all[fid])
            profile_batch.append(self.profile_all[fid])
            imgname_batch.append(self.imagename_all[fid])
            if self.cfg['use_flow']:
                for ft in flow_type:
                    flow_batch[ft].append(self.flow_all[ft][fid])
                    mask_flow_batch[ft].append(self.mask_flow_all[ft][fid])
                    mask_warp_batch[ft].append(self.mask_warp_all[ft][fid])
            if self.cfg['use_perceptual']:
                for l in range(len(self.cfg['perceptual_layers'])):
                    mask_per_batch[l].append(self.mask_per_all[l][fid])

       
        if self.cfg['use_flow']:
            for ft in flow_type:
                idx1, idx2 = (0, -1) if 'f' in ft else (1, self.batch_size)
                flow_batch[ft] = np.array(flow_batch[ft][idx1:idx2])
                mask_flow_batch[ft] = np.array(mask_flow_batch[ft][idx1:idx2])     
                mask_warp_batch[ft] = np.array(mask_warp_batch[ft][idx1:idx2])
        if self.cfg['use_perceptual']:
            for l in range(len(self.cfg['perceptual_layers'])):
                mask_per_batch[l] = np.array(mask_per_batch[l])     

        batch_data['cur_batch'] = cur_batch
        batch_data['batch_idx'] = batch_idx
        batch_data['batch_stride'] = batch_stride
        batch_data['input_batch'] = np.array(input_batch)
        batch_data['img_batch'] = np.array(img_batch)
        batch_data['mask_batch'] = np.array(mask_batch)
        batch_data['contour_batch'] = contour_batch
        batch_data['profile_batch'] = profile_batch
        batch_data['imgname_batch'] = imgname_batch
        
        if self.cfg['use_flow']:
            batch_data['flow_type'] = flow_type
            batch_data['flow_batch'] = flow_batch
            batch_data['mask_flow_batch'] = mask_flow_batch
            batch_data['mask_warp_batch'] = mask_warp_batch
            batch_data['flow_value_max'] = self.flow_value_max[batch_stride]
        if self.cfg['use_perceptual']:
            batch_data['mask_per_batch'] = mask_per_batch

        return batch_data


    def get_median_batch(self):
        return self.get_batch_data(int((self.cfg['frame_sum']) // 2), 1, ['f1'])


    def get_all_data(self):
        """
        Result a batch containing all the frames
        """
        batch_data = {}
        batch_data['input_batch'] = np.array(self.input_all[:self.frame_sum])
        batch_data['img_batch'] = self.image_all
        batch_data['mask_batch'] = self.mask_all
        batch_data['contour_batch'] = self.contour_all
        if self.cfg['use_perceptual']:
            batch_data['mask_per_batch'] = self.mask_per_all
        if self.cfg['use_flow']:
            batch_data['flow_type'] = self.cfg['flow_type']
            batch_data['flow_batch'] = self.flow_all
            batch_data['mask_flow_batch'] = self.mask_flow_all
            batch_data['mask_warp_batch'] = self.mask_warp_all
        return batch_data


    def load_single_frame(self, cap_video, cap_mask):
        gt = self.load_image(cap_video, False, self.frame_size)
        mask = self.load_image(cap_mask, True, self.frame_size)
        if self.cfg['dilation_iter'] > 0:
            mask = ndimage.binary_dilation(mask > 0, iterations=self.cfg['dilation_iter']).astype(np.float32)
        return gt, mask

    
    def load_image(self, cap, is_mask, resize=None):
        _, img = cap.read()
        if not resize is None:
            img = self.crop_and_resize(img, resize)
        if is_mask:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
            img = (img > 127) * 255
            img = img.astype('uint8')
        img_convert = img.transpose(2, 0, 1)
        return img_convert.astype(np.float32) / 255
    
    
    def crop_and_resize(self, img, resize, profile, interpolation=None):
        """
        Crop and resize img, keeping relative ratio unchanged
        """
        if interpolation is None:
            interpolation = self.cfg['interpolation']
        
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
        
        img = cv2.resize(img, (resize[1], resize[0]), interpolation=interpolation)
        
        if len(img.shape)!=3:
            img = np.expand_dims(img, axis=-1)
        
        img = img.transpose(2, 0, 1).astype(np.float32)
        return img, transform

    def polygon_to_binary_image(self, polygon, src):
        """
        Converts a Shapely polygon into a binary image with the same shape and coordinate system
        as the input .tif image.
    
        Parameters:
        polygon: The input polygon to rasterize.
        src: Image src.
    
        Returns:
        numpy.ndarray: Binary image array with the polygon rasterized.
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