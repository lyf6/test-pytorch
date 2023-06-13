from tifffile import imread, imsave
import numpy as np
import scipy.ndimage as ndi
from skimage import measure
from tqdm import tqdm
from PIL import Image
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

inter = 'whu_ocrnet_0.6'
prefix = '/home/yf/Documents/mmsegmentation/work_dirs/'+inter+'/pred/'
src_img ='/home/yf/disk/whu/part4/bcdd/two_period_data/change_label/change_label.tif'
 #prefix+'refine_diff.tif'
#'/home/yf/disk/whu/part4/bcdd/two_period_data/change_label/change_label.tif'
#prefix+'refine_diff.tif'
tar_img = prefix+'refine_diffv2.tif'
img1=np.array(imread(src_img))
tmp=np.zeros_like(img1)
index = img1>0
img1[index]=1
label=measure.label(img1,connectivity=2)
proper_tmp = measure.regionprops(label)
num_region = len(proper_tmp)
for re_id in tqdm(range(num_region)):
    area = proper_tmp[re_id].area
    perimeter = proper_tmp[re_id].perimeter
    score = perimeter/area
    print(perimeter, area, score)
    #area=proper_tmp[re_id].area
    if score<0.17:
        real_id = re_id+1
        index = label==real_id
        tmp[index] = 255


tif = gdal.Open(src_img)
prj = tif.GetProjection()
dtype = "Byte"
gt = tif.GetGeoTransform()
array2raster(tar_img, tmp, dtype, gt, prj)