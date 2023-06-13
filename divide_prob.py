import gdal, ogr, os, osr
import numpy as np
from math import *
import os
from tifffile import imread
from libtiff import TIFF as tiff


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



if __name__ == "__main__":

    imgpath_dir = '/home/yf/disk/whu/part4/bcdd/two_period_data/combine'
    target_dir = '/home/yf/disk/whu/part4/bcdd/two_period_data/5000/pred'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    weight = 5000
    height = 5000
    stride = 5000
    img_list = os.listdir(imgpath_dir)
    for path in img_list:
        imgpath = os.path.join(imgpath_dir, path)
        tif = gdal.Open(imgpath)
        prj = tif.GetProjection()
        dtype = "Float32"
        gt = tif.GetGeoTransform()
        x_min = gt[0]
        pixelWidth = gt[1]
        y_min = gt[3]
        pixelHeight = gt[5]
        b = gt[2]
        d = gt[4]
        img = np.array(tif.ReadAsArray())
        if img.ndim > 2:
            img = img.transpose((1,2,0))
        if img.ndim == 2:
            band_num = 1
            ori_w, ori_h= img.shape
        else:
            band_num = img.shape[2]
            ori_w, ori_h, _ = img.shape

        pad_w = (ceil(ori_w/stride))*stride-ori_w
        pad_h = (ceil(ori_h/stride))*stride-ori_h

        print(img.shape)
        if  band_num > 1:
            pad_img=np.pad(img,((0,pad_w),(0,pad_h),(0,0)),'constant', constant_values=0)
            #pad_img=np.zeros(())
            new_w, new_h,_= pad_img.shape
            # for rgb image, the non data is set as 0
            index = np.isnan(img)
            img[index] = 0
        else:
            pad_img=np.pad(img,((0,pad_w),(0,pad_h)),'constant', constant_values=255)
            new_w, new_h = pad_img.shape
            # for  image, the non data is set as 0
            index = np.isnan(img)
            img[index] = 0

        num_w = int(new_w/weight)
        num_h = int(new_h/height)
        for w_id in range(num_w):
            for h_id in range(num_h):
                if band_num > 1:
                    array=pad_img[w_id*stride:w_id*stride+weight, h_id*stride:h_id*stride+height,:]
                else:
                    array=pad_img[w_id*stride:w_id*stride+weight, h_id*stride:h_id*stride+height]
                print(pad_img)
                rasterOrigin = (x_min+w_id*stride*pixelWidth,y_min+h_id*stride*pixelHeight)
                originX = x_min+h_id*stride*pixelWidth
                originY = y_min+w_id*stride*pixelHeight
                geotransform = [originX, pixelWidth, b, originY, d, pixelHeight]
                #originX, pixelWidth, b, originY, d, pixelHeight = geotransform
                newRasterfn = os.path.join(target_dir, os.path.basename(imgpath).split('.')[0]+'_'+str(w_id)+'_'+str(h_id)+'.tif')
                array2raster(newRasterfn, array, dtype, geotransform, prj)
