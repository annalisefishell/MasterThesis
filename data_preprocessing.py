import numpy as np
import rasterio
import geopandas as gpd
import plotting as p
import ast

from shapely.geometry import box, mapping
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

WANTED_RES = 0.00018 # approx 20 meters

# Once per region
def find_largest_rectangle_geometry(polygon, steps=30, aspect_ratio=1.1):
    minx, miny, maxx, maxy = polygon.bounds
    best_rect = None
    best_area = 0

    x_vals = np.linspace(minx, maxx, steps)
    y_vals = np.linspace(miny, maxy, steps)

    for cx in x_vals:
        for cy in y_vals:
            # Binary search on width (since height = width * aspect_ratio)
            low, high = 0, min(maxx - minx, maxy - miny)
            while high - low > 1e-2:
                w = (low + high) / 2
                h = w * aspect_ratio  # enforce vertical orientation
                candidate = box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                if polygon.contains(candidate):
                    low = w
                    area = w * h
                    if area > best_area:
                        best_area = area
                        best_rect = candidate
                else:
                    high = w
    return best_rect


def shrink_polygon(poly, frac=1/40):
    minx, miny, maxx, maxy = poly.bounds
    dx = (maxx - minx) * frac
    dy = (maxy - miny) * frac * (6/5)
    return box(minx + dx, miny + dy, maxx - dx, maxy - dy)


def merge_rasters(list_of_files, output_path):
    if len(list_of_files)>1:
        # Open the two datasets
        src_files_to_mosaic = []
        for fp in list_of_files:
            src = rasterio.open(fp)
            src_files_to_mosaic.append(src)

        # Merge them
        mosaic, out_trans = merge(src_files_to_mosaic)

        # Get metadata from one of the input files
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "count": mosaic.shape[0]  # Number of bands
        })

        # Write to disk
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        print(f"Merged file saved to {output_path}")


def reproject_and_clip_raster(raster_path, geometry_gdf, output_path):
    # Read input raster
    with rasterio.open(raster_path) as src:
        # Get target CRS from geometry
        target_crs = geometry_gdf.crs
        
        # Calculate transform and dimensions for new resolution and CRS
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds, resolution=WANTED_RES
        )

        # Create destination array and metadata
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open('temp_reprojected.tif', 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )

    # Clip the reprojected raster to the geometry
    with rasterio.open('temp_reprojected.tif') as src:
        geoms = [mapping(geom) for geom in geometry_gdf.geometry]
        out_image, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Save final clipped raster
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)


# Reading the files
def get_aoi_geometry(file_path, need_resizing=False, save_to=''):
    if not need_resizing:
        aoi = gpd.read_file(file_path)['geometry']
    else:
        aoi = gpd.read_file(file_path)['geometry']
        aoi = gpd.GeoSeries([find_largest_rectangle_geometry(aoi[0])])
        aoi.to_file(save_to)
    return aoi


def open_tif_file(file_path):
    with rasterio.open(file_path) as src:
        image = src.read()
        profile = src.profile
    return image, profile


# Normalizing
def min_max_normalize(array):
    return (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))



# Read the data
def get_sentinel2(aoi, file_path, padding, temp_file='', list_files=[], plot=True):
    if len(list_files) > 0:
        if len(list_files) > 1:
            merge_rasters(list_files, temp_file)
        reproject_and_clip_raster(temp_file, aoi.geometry, file_path)

    else:
        s2_data, profile = open_tif_file(file_path)

        data = []
        for band in range(s2_data.shape[0]):
            vmin, vmax = np.percentile(s2_data[band], [0, 99]) #- cut top 1 percent for all bands
            temp = np.where(s2_data[band] <= vmax, s2_data[band], 0)
            data.append(min_max_normalize(temp))
        data = np.pad(np.array(data), padding, mode='constant')

        if plot:
            title = 'Sentinel 2 - Red Band ' + file_path.split('/')[-1].split('_')[2]
            p.plot_raster(data[3], title, 'BuGn')

        return data, profile


def get_interferogram(aoi, file_path, padding, original_path='', plot=True):
    if original_path != '':
        reproject_and_clip_raster(original_path, aoi.geometry, file_path)
    else:
        data, profile = open_tif_file(file_path)
        data = min_max_normalize(data)
        data = np.pad(data, padding, mode='constant')
        data = np.ma.masked_invalid(data)

        if plot:
            title = 'Sentinel 1 - Interferogram ' + '-'.join(file_path.split('/')[-1].split('_')[0:2])
            p.plot_raster(data[0], title, 'BuGn')

        return data, profile


def get_capella(aoi, file_path, padding, original_file='', plot=True):
    if original_file != '':
        reproject_and_clip_raster(original_file, aoi.geometry, file_path)
    else:
        data, profile = open_tif_file(file_path)

        vmin, vmax = np.percentile(data[0], [0, 99.5])
        data = np.where(data[0] <= vmax, data[0], np.nan)
        data = np.pad(data, padding, mode='constant')
        data = np.expand_dims(np.ma.masked_invalid(min_max_normalize(data)), axis=0)
        
        if plot:
            title = 'Capella - Backscatter ' + file_path.split('/')[-1].split('_')[1]
            p.plot_raster(data[0], title, 'BuGn')

        return data, profile


def get_iceye(aoi, file_path, padding, original_file='', plot=True):
    if original_file != '':
        reproject_and_clip_raster(original_file, aoi.geometry, file_path, WANTED_RES)
    else:
        data, profile = open_tif_file(file_path)

        vmin, vmax = np.percentile(data[0], [0, 99.5])
        data = np.where(data[0] <= vmax, data[0], np.nan)
        data = np.pad(data, padding, mode='constant')
        data = np.expand_dims(np.ma.masked_invalid(min_max_normalize(data)), axis=0)
        
        if plot:
            title = 'Iceye - Backscatter ' + file_path.split('/')[-1].split('_')[-2]
            p.plot_raster(data[0], title, 'BuGn')

        return data, profile


def get_terrasar(aoi, file_path, padding, original_file='', plot=True):
    if original_file != '':
        reproject_and_clip_raster(original_file, aoi.geometry, file_path, WANTED_RES)
    else:
        data, profile = open_tif_file(file_path)

        vmin, vmax = np.percentile(data[0], [0, 99.5])
        data = np.where(data[0] <= vmax, data[0], np.nan)
        data = np.pad(data, padding, mode='constant')
        data = np.expand_dims(np.ma.masked_invalid(min_max_normalize(data)), axis=0)
        
        if plot:
            p.plot_raster(data[0], 'TerraSAR - Polarizarion HH (23/05/2022)', 'BuGn')

        return data, profile


def get_ref_data(aoi, file_path, padding, original_path='', plot=True):
    if original_path != '':
        reproject_and_clip_raster(original_path, aoi.geometry, file_path)
    else:
        data, profile = open_tif_file(file_path)

        data = np.expand_dims(np.pad(data[0], padding, mode='constant'), axis=0)

        if plot:
            p.plot_raster(data[0], 'Woody AGB - Harris et al. 2021', 'BuGn', 
                          normalized=False, cbar_label='Mg/ha')
            
        return data, profile
    

def extract_patches(img, patch_size, overlap=0):
    h, w = img.shape[:2]
    step = patch_size - overlap
    patches = []
    for i in range(0, h - patch_size + 1, step):
        for j in range(0, w - patch_size + 1, step):
            patches.append(img[i:i+patch_size, j:j+patch_size, :])
    return np.array(patches)
        

def get_test_train_data(combined, n_pixels, n_bands, ref_data, test_percent=0.3, cnn=False, 
                        patch_size=200, overlap=0):
    if not cnn:
        X_2d = np.transpose(combined, (1, 2, 0)).reshape(n_pixels, n_bands)
        y_1d = ref_data.reshape(n_pixels)

        # Impute missing values in X (mean strategy)
        imputer = SimpleImputer(strategy='mean')
        X_2d = imputer.fit_transform(X_2d)

        # Split into test and train 
        X_train, X_test, y_train, y_test = train_test_split(
            X_2d, y_1d, test_size=test_percent, random_state=42
        )

        return X_train, X_test, y_train, y_test, X_2d
    else:
        all_img_patches = []
        all_target_patches = []
        for img in combined:
            img = np.expand_dims(img, axis=2)
            img_p = extract_patches(img, patch_size, overlap)
            all_img_patches.append(img_p)
        all_img_patches = np.array(all_img_patches)
        all_img_patches = np.nan_to_num(all_img_patches, nan=0.0, posinf=0.0, neginf=0.0)

        all_target_patches = extract_patches(np.expand_dims(ref_data[0], axis=2), patch_size, overlap)
        all_target_patches = np.nan_to_num(all_target_patches, nan=0.0, posinf=0.0, neginf=0.0)

        total = all_img_patches.shape[1]
        indices = np.arange(total)
        np.random.shuffle(indices)
        split = int(test_percent * total)
        train_idx, test_idx = indices[:split], indices[split:]

        train_imgs = []
        test_imgs = []
        for img in all_img_patches:
            temp_train, temp_test = img[train_idx], img[test_idx]
            train_imgs.append(temp_train)
            test_imgs.append(temp_test)

        train_imgs = np.squeeze(train_imgs, axis=-1)  
        train_imgs = train_imgs.transpose(1, 2, 3, 0)

        test_imgs = np.squeeze(test_imgs, axis=-1)  
        test_imgs = test_imgs.transpose(1, 2, 3, 0)

        train_tgts, test_tgts = np.array(all_target_patches[train_idx]), all_target_patches[test_idx]

        return train_imgs, test_imgs, train_tgts, test_tgts, all_target_patches
    

def string_to_dict(s):
    data = ast.literal_eval(s.strip())
    return data

def remove_item(list, item):
    res = [i for i in list if i != item]
    return res

def combine_dicts(key, main_dict, add_dict):
    if key in main_dict:
        old_val = main_dict[key]

        for k in old_val.keys():
          old_list = old_val[k]
          new_list = add_dict[k] 
          new_val = []
          for i in range(len(old_list)):
            merged = old_list[i] + new_list[i]
            new_val.append(merged)

          old_val[k] = new_val

        main_dict.update({key: old_val})
    else:
        main_dict.update({key: add_dict})
    return main_dict

def read_eval_file(file_path):
    file = open(file_path, 'r').readlines()
    file = remove_item(file, '\n')

    results = {}
    for i in range(0, len(file) - 1, 2):
        key = file[i].strip().strip(':').split()
        key = key[0] + ' ' + key[-1]

        val = string_to_dict(file[i+1])

        results = combine_dicts(key, results, val)
    return results

def get_metrics(my_dict, key):
    if type(my_dict[key])!=dict:
        metric_list = list(my_dict[key])
    else:
        metric_list = list(my_dict[key].values())[0]

    run_time = np.mean(metric_list[0])
    rmse = np.mean(metric_list[1])
    r2 = np.mean(metric_list[2])

    return run_time, rmse, r2

def get_mean_metrics(main_dict, key):
    lst = main_dict[key]

    times = []
    rmses = []
    r2s = []

    for k in lst:
        t, rmse, r2 = get_metrics(lst, k)
        times.append(t)
        rmses.append(rmse)
        r2s.append(r2)

    return times, rmses, r2s