import numpy as np
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import gdal, osr

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

def _mask2rgb(mask, colormap):
    w,h =mask.shape
    img_mask=np.ones(shape=(w,h,3),dtype=np.uint8)*(0)
    for id, rgb in enumerate(colormap):
        #index=rgbmask==rgb
        index=(mask == id)
        #index = zip(rgbmask ==rgb)
        color=colormap[id]
        img_mask[index]=color
    return img_mask.copy()

# inter = 'gid_fcn'
# dir='/home/yf/Documents/mmsegmentation/work_dirs/'+inter+'/pred'
# target='/home/yf/disk/tmp/'+inter

target = '/home/yf/disk/res/rgb'
dir = '/home/yf/disk/res/combine'
files=['GF2_PMS2__L1A0001433318-MSS2pred.tif', 'GF2_PMS1__L1A0001734328-MSS1pred.tif']
# colormap=[[0, 200, 0],[150, 250, 0],
#            [150, 200, 150], [200, 0, 200],
#            [150, 0, 250], [150, 150, 250],
#            [250, 200, 0], [200, 200, 0],
#            [200, 0, 0], [250, 0, 150],
#            [200, 150, 150], [250, 150, 150],
#            [0, 0, 200], [0, 150, 200],
#            [0, 200, 250], [0, 0, 0]] #color map for sparse compete
colormap=[[255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 255, 0], [0, 0, 255],[0, 0, 0]]  # color map for gid
#colormap = [[255,0,0],[0,255,0],[0,0,0]]
# colormap=[[0, 200, 0],[150, 250, 0],
#            [150, 200, 150], [200, 0, 200],
#            [150, 0, 250], [150, 150, 250],
#            [250, 200, 0], [200, 200, 0],
#            [200, 0, 0], [250, 0, 150],[0, 150, 200]] #color map for tianchi
if not os.path.exists(target):
    os.makedirs(target)

for file in files:
    path=os.path.join(dir,file)
    tif = gdal.Open(path)
    prj = tif.GetProjection()
    dtype = "Byte"
    gt = tif.GetGeoTransform()
    #img=np.array(Image.open(path))
    img =  np.array(tif.ReadAsArray())
    mask=_mask2rgb(img, colormap)
    #imgpath = os.path.join(mask_dir,file.split(''))
    target_path = os.path.join(target, file)
    array2raster(target_path, mask, dtype, gt, prj)
    # mask=Image.fromarray(mask)
    # target_path=file.split('.pn')[0]+'.png'
    # target_path=os.path.join(target,target_path)
    # mask.save(target_path)