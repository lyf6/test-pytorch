from __future__ import division
import numpy as np
import os
from PIL import Image
from math import *
import gdal, ogr, os, osr
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





t1 = '/home/yf/Documents/test-pytorch/crop2017_class.tif'
t2 = '/home/yf/Documents/test-pytorch/crop2019_class.tif'
save_path = '/home/yf/Documents/test-pytorch/diff.tif'


tif = gdal.Open(t1)
prj = tif.GetProjection()
dtype = "Byte"
gt = tif.GetGeoTransform()
# x_min = gt[0]
# pixelWidth = gt[1]
# y_min = gt[3]
# pixelHeight = gt[5]
# b = gt[2]
# d = gt[4]
t1_img = np.array(tif.ReadAsArray())
# t1_img = t1_img.transpose((1,2,0))

tif = gdal.Open(t2)
t2_img = np.array(tif.ReadAsArray())
# t2_img = t2_img.transpose((1,2,0))

tmp = np.zeros_like(t2_img)
index = t1_img != t2_img
tmp[index] = 255
array2raster(save_path, tmp, dtype, gt, prj)
