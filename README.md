# DIL_SDS
Use Deep Internal Learning (DIL) to reconstruct cloud-contaminated images and apply satellite derived shoreline

## 1.Get started
### 1.1 Install coastsat
Install CoastSat Toolbox following the instruction [here](https://github.com/kvos/CoastSat?tab=readme-ov-file#installation).
### 1.2 Install other required packages
```
conda install -c conda-forge rasterio 
```

## 2.Usage
### 2.1 Download images
To retrieve from the GEE server the available satellite images cropped around the user-defined region of coastline for the particular time period of interest, the following variables are required in `download_configs/download_config.json`:\
`AOI_PATH`: Path to the user-defined area of interest (AOI), an example is provided in `data/Narrabeen/AOI.geojson`.\
`AOI_NAME`: Name (e.g. Narrabeen) for the AOI.\
`OPTICAL_MISSION`: Landsat missions to include (e.g. ["L8", "L9"] representing Landsat 8 and Landsat 9).\
`STAET_DATE`: Starting data for image retrieving (e.g. "2017-01-01").\
`END_DATE`: Starting data for image retrieving (e.g. "2023-01-01").\
`S1_MOSAIC_WINDOW`: Time window (in days) to mosaic Sentinel 1 images.\
`PROCESS_STAGE`: Stages of optical images to use ("TOA"|"SR"), choose between Top Of Atmosphere (TOA) or Surface Reflectance (SR).\
`OUT_DIR`: Directory to save image outputs.\

To download Landsat images:
```
python src\optical_download.py
```
To download Sentinel 1 images
```
python src\S1_download.py
```
### 2.2 Preprocess images (pairing, warp, and cloud synthesizing)
Run preprocess.py to pair, warp and superimpose clouds to clear optical images. The following arguements are required:
`--data_path`: The path of previously downloaded data (The same as `OUT_DIR` in download_config.json).
`--cloud_ratio`: The ratio of cloud to superimpose, ranging between 0 and 1
`--temporal_var`: Stores true value. Adding this arguement will add seasonallity to synthetic clouds.
```
python src/preprocess.py --data_path {} --cloud_ratio {}
```
- More parameters to config can be found with
```
python src/preprocess.py -h
```


### 2.3 Run DIL model for image reconstruction
### 2.4 Run CoastSat for shoreline extraction
