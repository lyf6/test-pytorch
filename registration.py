from arosics import COREG_LOCAL
import os
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







rasterLayers_1='/home/yf/disk/2017-2019/2017v1.tif'
rasterLayers_2='/home/yf/disk/2017-2019/2019.tif'


tif = gdal.Open(rasterLayers_1, gdal.GA_Update)
prj = tif.GetProjection()
print(prj)
sr = osr.SpatialReference()
sr.ImportFromEPSG(4326)
tif.SetProjection(sr.ExportToWkt())
tif = gdal.Open(rasterLayers_2, gdal.GA_Update)
tif.SetProjection(sr.ExportToWkt())


grid_res = 50
window_size = 256
workdir = '/home/yf/disk/2017-2019/workdir'
out_name= os.path.join(workdir,'2019re.tif')
max_shift = 5
min_reliability = 0.6
max_iter = 20
kwargs = {
    'grid_res': grid_res,
    'window_size': (window_size, window_size),
    'fmt_out': 'GTIFF',
    'path_out': 'auto',
    'projectDir': workdir,
    'q': False,
    'align_grids': True,
    'match_gsd': True,
    'path_out': out_name,
    'max_shift': max_shift,
    'min_reliability': min_reliability,
    'max_iter': max_iter
}
CRL = COREG_LOCAL(rasterLayers_1,rasterLayers_2,**kwargs)
CRL.correct_shifts()
path_out = os.path.join(workdir, 'registration_tie_point.shp')
CRL.tiepoint_grid.to_PointShapefile(path_out=path_out)