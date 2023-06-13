from __future__ import division
import numpy as np
import os
from PIL import Image
from math import *
import gdal, ogr, os, osr
from tifffile import imread

Image.MAX_IMAGE_PIXELS = None


def array2raster(newRasterfn, array, dtype, geotransform, prj):
    """
    save GTiff file from numpy.array
    input:
        newRasterfn: save file name
        dataset : original tif file
        array : numpy.array
        dtype: Byte or Float32.
    """
    cols = array.shape[1]
    rows = array.shape[0]
    originX, pixelWidth, b, originY, d, pixelHeight = geotransform

    driver = gdal.GetDriverByName('GTiff')

    # set data type to save.
    GDT_dtype = gdal.GDT_Unknown
    if dtype == "Byte":
        GDT_dtype = gdal.GDT_Byte
    elif dtype == "Float32":
        GDT_dtype = gdal.GDT_Float32
    else:
        print("Not supported data type.")

    # set number of band.
    if array.ndim == 2:
        band_num = 1
    else:
        band_num = array.shape[2]

    outRaster = driver.Create(newRasterfn, cols, rows, band_num, GDT_dtype)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

    # Loop over all bands.
    for b in range(band_num):
        outband = outRaster.GetRasterBand(b + 1)
        # Read in the band's data into the third dimension of our array
        if band_num == 1:
            outband.WriteArray(array)
        else:
            outband.WriteArray(array[:,:,b])

    outRasterSRS = osr.SpatialReference(wkt=prj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()


# dir='/home/yf/disk/myannotated_cd/rgbp'
# target_dir='/home/yf/disk/myannotated_cd/prod_combv2'
# result_path= '/home/yf/disk/myannotated_cd/rgbp-res'

filter_dir='/home/yf/disk/gid/test/masks'
#'/home/yf/disk/myannotated_cd/2017-2019val/imgs'
# '/home/yf/disk/whu/part4/bcdd/two_period_data/before/image'
target_dir= '/home/yf/Documents/mmsegmentation/work_dirs/gid_fcn/pred'
#'/home/yf/disk/myannotated_cd/prod_comb'
#'/home/yf/disk/whu/part4/bcdd/two_period_data/combine'
#result_path= '/home/yf/Documents/mmsegmentation/work_dirs/whu_pspnet/results'
#'/home/yf/disk/myannotated_cd/test_masks'
    #'/home/yf/disk/output/myann/seg_hrnet_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484x2/test_results'
files=os.listdir(target_dir)
weight=512
height=512
stride=512




for file in files:
    #if file=='bdf.tif':
    #path=os.path.join(dir,file)
    #print(path)
    imgpath = os.path.join(target_dir, file)
    print(imgpath)
    tif = gdal.Open(imgpath)
    prj = tif.GetProjection()
    dtype = "Byte"
    gt = tif.GetGeoTransform()
    # x_min = gt[0]
    # pixelWidth = gt[1]
    # y_min = gt[3]
    # pixelHeight = gt[5]
    # b = gt[2]
    # d = gt[4]
    filter_path = os.path.join(filter_dir,file.split('pred')[0]+'_label.tif')
    print(filter_path)
    tar_img = np.array(tif.ReadAsArray())
    fileter_img = np.array(gdal.Open(filter_path).ReadAsArray())
    index = fileter_img==5
    #print(index)
    tar_img[index]=5
   
    array2raster(imgpath, tar_img, dtype, gt, prj)
