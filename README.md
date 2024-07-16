# DIL_SDS
This project applies a Deep Internal Learning ([DIL](https://github.com/Haotianz94/IL_video_inpainting)) to reconstruct cloud-contaminated MNDWI based on Landsat 8&9 images and extracts shoreline positions from reconstructed images with [CoastSat Toolbox](https://github.com/kvos/CoastSat). Different from the original DIL using either Gaussian noise as priors, the proposed method used the mixture of the Gaussian noise and Sentinel 1 SAR images based on the availability of SAR images. Additionally, a shoreline-focus loss function was introduced in this project to optimize the MNDWI reconstruction only in the 200 m buffer of the shoreline. 


## 1.Install
### 1.1 Creat coastsat env to download images and extract shorelines
Install CoastSat Toolbox following the instruction [here](https://github.com/kvos/CoastSat?tab=readme-ov-file#installation).
In addition to standard CoastSat Toolbox:
```
conda install -c conda-forge rasterio
```
### 1.2 Create DIL env to reconstruct cloud-contaminated images

The DIL code has been tested on pytorch 1.13.1 with python 3.7 and cuda 11.7. Please refer to [requirements.txt](https://github.com/yongjingmao/DIL_SDS/blob/main/requirements.txt) for details.
```
conda create -n DIL python=3.7
conda activate DIL
pip install -r requirements.txt
```

## 2.Usage
Implementing DIL_SDS requires files below:\
`AOI.geojson`: A polygon define the area of interest (AOI).\
`transects.geojson`: A sequence of shore-normal transects to define shoreline position.\
`ref_shoreline.geojson`: A reference shoreline to define area of focus in image reconstruction.\
Examples of above files were provided for Narrabeen, Coolangatta, Ocean Beach, and Castelldefels in `data`.

### 2.1 Download images
To retrieve from the GEE server the available satellite images cropped around the user-defined region of coastline for the particular time period of interest, the following variables are required in `download_configs/download_config.json`:\
`AOI_PATH`: Path to the user-defined area of interest (AOI), an example is provided in `data/Narrabeen/AOI.geojson`.\
`OUT_DIR`: Directory to save image outputs.\
`AOI_NAME`: Name (e.g. Narrabeen) for the AOI.\
`OPTICAL_MISSION`: Landsat missions to include (e.g. ["L8", "L9"] representing Landsat 8 and Landsat 9).\
`STAET_DATE`: Starting data for image retrieving (e.g. "2017-01-01").\
`END_DATE`: Starting data for image retrieving (e.g. "2023-01-01").\
`S1_MOSAIC_WINDOW`: Time window (in days) to mosaic Sentinel 1 images.\
`PROCESS_STAGE`: Stages of optical images to use ("TOA"|"SR"), choose between Top Of Atmosphere (TOA) or Surface Reflectance (SR).

To download Landsat images:
```
conda activate coastsat
python src/optical_download.py
```
To download Sentinel 1 images
```
python src/S1_download.py
```
- SAR, Optical images and Optical Masks will be downloaded to subfolders `SAR`, `Optical` and `Mask` in `OUT_DIR/S1_Landsat` respectively.

### 2.2 Preprocess images (pairing, warp, and cloud synthesizing)
Run `preprocess.py` to pair, warp and superimpose clouds to clear optical images. The following arguements are required:\
`--data_path`: The path of previously downloaded data (The same as `OUT_DIR`).\
`--cloud_ratio`: The ratio of cloud to superimpose, ranging between 0 and 1.\
`--temporal_var`: Stores true value. Adding this arguement will add seasonallity to synthetic clouds.

```
conda activate coastsat
python src/preprocess.py --data_path data/Narrabeen  --cloud_ratio 0.5
```
- Resultant synthetic images of optical, mndwi and cloud mask are saved in subfolders `Optical_50`, `MNDWI_50`and `Mask_50` in `data_path/S1_Landsat` respectively

### 2.3 Run DIL model for image reconstruction
Use `DIL_run.py` to reconstruct cloud contaminated images. The DIL model requires a reference shoreline and an exaple is provided as `data/Narrabeen/ref_shoreline.geojson`.\
The following arguements are required:\
`--data_path`: The same as `data_path` in preprocess.\
`--res_dir`: Directory to save model outputs.\
`--train_mode`: Type of DIL model to use. Choose among DIP|DIP_Vid|DIP_Vid_3DCN|DIP_Vid_Flow
`--resize`: Tuples include Height and width of the output. Both must be the multiples of 64. Input images will be clipped and resampled to fit the resize.\
`--batch_size`: Batch size of DIL model. batch_size = 5 is the optimum value for the example Narrabeen site.\
`--input_type`: Type of input priors. Choose between 'S1'|'Noise'.\
`--cloud_ratio`: The same as `cloud_ratio` in preprocess.\
`--num_pass`: Number of passes for DIL model.\
`--paired`: Stores true value. Adding this arguement will only include optical images paired with SAR.
```
conda activate DIL
python src/DIL_run.py --data_path data/Narrabeen --res_dir results/Narrabeen --train_mode DIP-Vid-3DCN --resize 384 192 --batch_size 5 --input_type S1 --cloud_ratio 0.5 --num_pass 10 
```
- More parameters of DIL model itself can be tuned in the config files in `DIL/configs`.
- Mdel result of each pass will be saved in `res_dir` as an individual subfolder.

### 2.4 Run CoastSat for shoreline extraction
Run `shoreline_extraction.py` to extract shorelines based on target and modelled MNDWI images. The following arguements are required:\
`--data_path`: The same as `data_path` in preprocess.\
`--result_path`: The same as `res_dir` in preprocess.\
`--cloud_ratio`: The same as `cloud_ratio` in preprocess.\
`--num_pass`: The same as `num_pass` in DIL_run.\
`--max_dist_ref`: Maximum allowed distance (m) to reference shoreline.\
`--min_length`: Minimum allowed length (m) of shoreline segments.\
```
conda activate coastsat
python src/shoreline_extraction.py --data_path data/Narrabeen --result_path results/Narrabeen --cloud_ratio 0.5 --num_pass 10 --max_dist_ref 200 --min_length 50
```
- Target shoreline positions are saved as `SDS.csv` in `data_path/S1_Landsat`; modelled shorelne positions under and outside of synthetic clouds are saved as `SDS_cloud.csv` and `SDS_clear.csv` in `result_path`.

## Acknowledgement
The implementation of the DIL network architecture is mostly borrowed from the [IL_video_inpainting](https://github.com/Haotianz94/IL_video_inpainting/tree/master). The shoreline extraction is mostly based on the [CoastSat Toolbox](https://github.com/kvos/CoastSat/tree/master). Should you be making use of this work, please make sure to adhere to the licensing terms of the original authors.
