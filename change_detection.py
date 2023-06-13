from tifffile import imread, imsave
import numpy as np
import cupy as cp
import scipy.ndimage as ndi
from skimage import measure
from tqdm import tqdm
from PIL import Image
import gdal, ogr, os, osr
from tifffile import imread
import cv2
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


time1_series = ['baodi20120815', 'baodi20150918', 'jinghai120110618', 'jinghai120131108',\
                'jinghai220110618', 'jinghai220140529', 'jinnan20110511', 'jinnan20150913',\
                'ninghe20120617','ninghe20150507']
time2_series = ['baodi20150918', 'baodi20190302', 'jinghai120131108','jinghai120190613' ,\
                'jinghai220140529', 'jinghai220200519', 'jinnan20150913', 'jinnan20181018',\
                'ninghe20150507', 'ninghe20200519']
pred_dir = '/home/yf/disk/res/objs'
diff_dir = '/home/yf/disk/res/diff_dir'
values = [-1, 1]
for time1_name, time2_name in zip(time1_series, time2_series):
    time2_path= os.path.join(pred_dir,time2_name + 'pred.tif')
    time1_path=os.path.join(pred_dir,time1_name + 'pred.tif')
    diff_path=os.path.join(diff_dir, time2_name + '_' + time1_name +'.tif')
    img1=np.asarray(imread(time1_path), dtype=np.int)
    img2=np.asarray(imread(time2_path), dtype=np.int)
    diff=img2-img1
    result = np.zeros_like(diff)
    for value in values:
        tmp=diff.copy()
        one_index = tmp == value
        zero_index = tmp != value
        tmp[one_index] = 1
        tmp[zero_index] = 0 
        tmp = tmp.astype(np.uint8)       
        kernel = np.ones((15,15),np.uint8)
        tmp = cv2.erode(tmp,kernel)
        tmp_label = measure.label(tmp, connectivity=2)
        proper_tmp = measure.regionprops(tmp_label)
        num_region = len(proper_tmp)
        for re_id in tqdm(range(num_region)):
            sub_region_aera =  proper_tmp[re_id].area
            if sub_region_aera>40000:
                real_id = re_id+1
                index = tmp_label==real_id
                result[index]=1

        tif = gdal.Open(time2_path)
        prj = tif.GetProjection()
        dtype = "Byte"
        gt = tif.GetGeoTransform()
        array2raster(diff_path, result, dtype, gt, prj)


