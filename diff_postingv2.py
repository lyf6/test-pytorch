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



# img1_path='/home/yf/disk/qgis34-files/object/after_pred.tif'
# img2_path='/home/yf/disk/qgis34-files/object/before_pred.tif'
# diff_path='/home/yf/disk/qgis34-files/object/refine_diff.tif'
prefix = '/home/yf/Documents/mmsegmentation/work_dirs/whu_pspnet/pred'
img1_path= os.path.join(prefix,'afterpred.tif')
#'/home/yf/disk/whu/part4/bcdd/two_period_data/combine_2000/afterpred.tif'
#'/home/yf/disk/qgis34-files/object/after_pred.tif'
img2_path=os.path.join(prefix,'beforepred.tif')
#'/home/yf/disk/whu/part4/bcdd/two_period_data/combine_2000/beforepred.tif'
#'/home/yf/disk/qgis34-files/object/before_pred.tif'
diff_path=os.path.join(prefix,'refine_diff.tif')
#'/home/yf/disk/whu/part4/bcdd/two_period_data/combine_2000/refine_diff.tif'
#'/home/yf/disk/qgis34-files/object/refine_diff.tif'
img1=np.array(imread(img1_path))
img2=np.array(imread(img2_path))
label1=measure.label(img1,connectivity=2)
label2=measure.label(img2,connectivity=2)
diff=img1-img2
proper_1 = measure.regionprops(label1)
proper_2 = measure.regionprops(label2)
collect=[1,-1]
result = np.zeros_like(diff)
for val in collect:
    tmp=diff.copy()
    tmp[tmp != val] = 0
    tmp[tmp == val] = 1
    # tmp_label=measure.label(tmp,connectivity=2)
    # regin_ids = np.unique(tmp_label)
    if val == 1:
        label = label1
        proper = proper_1
        value = 255
    else:
        label = label2
        proper = proper_2
        value = 255
    tmp_label = measure.label(tmp, connectivity=2)
    proper_tmp = measure.regionprops(tmp_label)
    num_region = len(proper_tmp)
    for re_id in tqdm(range(num_region)):
        sub_region_aera =  proper_tmp[re_id].area
        centroid = proper_tmp[re_id].coords[0]
        #print(centroid)
        region_id = label[centroid[0],centroid[1]]-1
        #print(region_id)
        area = proper[region_id].area
        if sub_region_aera/area>0.4 or sub_region_aera>5000:
            real_id = re_id+1
            index = tmp_label==real_id
            result[index]=value

tif = gdal.Open(img1_path)
prj = tif.GetProjection()
dtype = "Byte"
gt = tif.GetGeoTransform()
array2raster(diff_path, result, dtype, gt, prj)


