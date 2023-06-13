from __future__ import division
import numpy as np
import os
from PIL import Image
from math import *
import gdal, ogr, os, osr
from tifffile import imread
from libtiff import TIFF as tiff
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



dir='/home/yf/disk/whu/part4/bcdd/two_period_data/after/image'
target_dir='/home/yf/disk/whu/part4/bcdd/two_period_data/combine_pred'
result_path= '/home/yf/disk/whu/part4/bcdd/two_period_data/results'
    #'/home/yf/disk/output/myann/seg_hrnet_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484x2/test_results'
files=os.listdir(dir)
weight=512
height=512
stride=512

for file in files:
    #if '2019-test.tif' in file or '2017-test.tif' in file:
    path=os.path.join(dir,file)
    imgpath = os.path.join(dir, path)
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

    img = np.array(tif.ReadAsArray())
    # print(img.shape)
    img = img.transpose((1,2,0))
    # img = np.array(Image.open(path))
    ori_w, ori_h, c = img.shape
    #rint(ori_w, ori_h, c)



    num_w = int(new_w/weight)
    num_h = int(new_h/height)
    print(num_w)
    for w_id in range(num_w):
        for h_id in range(num_h):
            tmp_path = file.split('.tif')[0]  + '_'+str(w_id)+'_'+str(h_id) + '.tif'
            print(tmp_path)
            tmp_path = os.path.join(result_path,tmp_path)
            tmp_img = np.array(imread(tmp_path))
            tmp_img = tmp_img.transpose((1,2,0))
                # Image.open(tmp_path)
            pad_img[ w_id * stride:w_id * stride + weight, h_id * stride:h_id * stride + height] = tmp_img
                                                                                                    #*255).astype(np.int)
            #print(pad_img)
            # count=count+1
    pad_img=pad_img[:ori_w,:ori_h]
    pad_img=np.argmax(pad_img, axis=2)
    #print(pad_img)
    target_path = file.split('.ti')[0] +  '.tif'
    target_path = os.path.join(target_dir,target_path)
    array2raster(target_path, pad_img, dtype, gt, prj)
