from osgeo import gdal, osr
import cv2
import numpy as np
from PIL import Image
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
    #originX, pixelWidth, b, originY, d, pixelHeight = geotransform

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
    outRaster.SetGeoTransform(geotransform)

    # Loop over all bands.
    for b in range(band_num):
        outband = outRaster.GetRasterBand(b + 1)
        # Read in the band's data into the third dimension of our array
        if band_num == 1:
            outband.WriteArray(array)
        else:
            outband.WriteArray(array[:, :, b])

    outRasterSRS = osr.SpatialReference(wkt=prj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

dsm = "/home/yf/disk/jinzhongqiao/Production_3_ortho_merge_scale.tif"
dom = "/home/yf/disk/jinzhongqiao/Production_3_ortho_merge_scale.tif"
save_path = "/home/yf/disk/jinzhongqiao/dom_dsm.tif"

dom_img_handle = gdal.Open(dom)
geotransform = dom_img_handle.GetGeoTransform()
prj = dom_img_handle.GetProjection()
dom_img = dom_img_handle.ReadAsArray()

dsm_img = Image.open(dsm)
dsm_img = np.array(dsm_img, dtype=np.uint8)
# print(4)
dom_img[2, :, :] = dsm_img
dtype = "Byte"
dom_img = dom_img.transpose((1,2,0))

array2raster(save_path, dom_img, dtype, geotransform, prj)
