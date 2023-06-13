from __future__ import division
import numpy as np
import os
from PIL import Image
from math import *
import gdal, ogr, os, osr
from tifffile import imread
import gc
from multiprocessing import Pool, RawArray
from shutil import copyfile
import ctypes
from collections import Counter
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

var_dict_clsmap = {}
def init_worker(segmataion_map, predict_map):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict_clsmap['segmataion_map'] = segmataion_map
    var_dict_clsmap['predict_map'] = predict_map

def obtain_clsmap_object_based(seg_id):

    print(seg_id)
    segmataion_map = var_dict_clsmap['segmataion_map']

    shape = segmataion_map.shape
    predict_map = np.frombuffer(var_dict_clsmap['predict_map'], dtype=np.uint8).reshape(shape)
    tmp_map = segmataion_map == seg_id
    seg_pred = predict_map[tmp_map]
    max_occur_label = np.argmax(np.bincount(seg_pred))
    predict_map[tmp_map] = max_occur_label
    return predict_map

def obtain_clsmap_object_based_parrel(predict_map, segmataion_map):

    X_shape = predict_map.shape
    seglist = np.unique(segmataion_map)
    X = RawArray(ctypes.c_uint8, X_shape[0] * X_shape[1])
    X_n = np.frombuffer(X,dtype=np.uint8).reshape(X_shape)
    np.copyto(X_n, predict_map)
    ####
    pool = Pool(processes=10, initializer=init_worker, initargs=(segmataion_map, X))
    pool.map(obtain_clsmap_object_based, seglist)
    pool.close()
    pool.join()
    del pool
    gc.collect()
    out_data = np.array(X_n)
    return out_data
# def Most_Common(lst):
#     data = Counter(lst)
#     return data.most_common(1)[0][0]
#
# def ocnn(img_seg, width, height, pre_large_im):
#     # Convert the segment indices from 1...n
#     img_statics = np.zeros((width, height),dtype=np.int)
#     temp_unique = np.unique(img_seg)
#     for p in range(len(temp_unique)):
#         print(p,len(temp_unique))
#         temp_value = temp_unique[p]
#         idx = np.where(img_seg==temp_value)
#         temp_gt = pre_large_im[idx[0],idx[1]]
#         temp_label = np.int(Most_Common(temp_gt))
#         img_statics[idx[0],idx[1]] = temp_label
#     return img_statics

ori_cls='/home/yf/disk/myannotated_cd/prod_comb/2019-test_pred.tif'
result_path='/home/yf/disk/myannotated_cd/prod_comb/2019-test_pred_refine_seg.tif'
seg_path = '/home/yf/disk/myannotated_cd/ori_test/2017-segmentation.tif'

cls = gdal.Open(ori_cls)
prj = cls.GetProjection()
dtype = "Byte"
gt = cls.GetGeoTransform()
cls = np.array(cls.ReadAsArray())
#cls = cls.transpose((1,2,0))

seg = gdal.Open(seg_path)
seg = np.array(seg.ReadAsArray())
#seg = seg.transpose((1,2,0))
w, h = seg.shape
cls = cls[:w, :h]

#img_statics =  obtain_clsmap_object_based_parrel(cls, seg)
    #ocnn(seg, w, h, cls)
array2raster(result_path, cls, dtype, gt, prj)