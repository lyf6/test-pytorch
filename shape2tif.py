# -*- coding: UTF-8 -*-
import ogr, osr
import gdal
import numpy as np
import sys
import geopandas as gpd
# import rasterio
# import fiona
# from rasterio import features
import os
# from datetime import datetime as dt
from PIL import Image, ImageDraw

import pandas as pd

def multi2single(gpdf):
    gpdf_singlepoly = gpdf[gpdf.geometry.type == 'Polygon']
    gpdf_multipoly = gpdf[gpdf.geometry.type == 'MultiPolygon']

    for i, row in gpdf_multipoly.iterrows():
        Series_geometries = pd.Series(row.geometry)
        df = pd.concat([gpd.GeoDataFrame(row, crs=gpdf_multipoly.crs).T]*len(Series_geometries), ignore_index=True)
        df['geometry']  = Series_geometries
        gpdf_singlepoly = pd.concat([gpdf_singlepoly, df])

    gpdf_singlepoly.reset_index(inplace=True, drop=True)
    return gpdf_singlepoly

def shape2mask(raster, shape, valField, ignore_label):
    dataset = gdal.Open(raster)
    col = dataset.RasterXSize
    row = dataset.RasterYSize
    geotransform = dataset.GetGeoTransform()
    prj = dataset.GetProjection()
    ulY, ulX = geotransform[3], geotransform[0]
    distY, distX = abs(geotransform[5]), abs(geotransform[1])
    layer = gpd.read_file(shape)
    layer = multi2single(layer)
    rasterPoly = Image.new("I", (col, row), ignore_label)
    rasterize = ImageDraw.Draw(rasterPoly)
    # for geo in geometry:
    #     geo['area'] = geo.area

    if layer.empty:
        mask = np.array(rasterPoly)*0
        return mask, geotransform, prj

    layer['area'] = layer.apply(lambda row: row.geometry.area, axis=1)
    layer.sort_values(by='area', ascending=False, inplace=True)
    geometry = layer.geometry
    class_id = layer[valField]
    for geo, class_label in zip(geometry,class_id):
        # print(geo)
        #if class_label != 0:
        pixels = []
        xs, ys = geo.exterior.coords.xy
        class_label = int(1)
        for x, y,  in zip(xs, ys):
            pixel_x = abs(int((x - ulX) / distX))
            pixel_y = abs(int((y - ulY) / distY))
            pixel_y = row if pixel_y > row else pixel_y
            pixel_x = col if pixel_x > col else pixel_x
            pixels.append((pixel_x, pixel_y))
        #print(pixels)
        rasterize.polygon(pixels, fill=class_label)
    mask = np.array(rasterPoly)
    return mask, geotransform, prj

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
    print(cols, rows)
    outRaster.SetGeoTransform(geotransform)

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

def write_tif(rasterfile, shaplefile, outrasterfile, valField, ignore_label):

    mask, geotransform, prj = shape2mask(rasterfile, shaplefile, valField, ignore_label)
    dtype = "Byte"
    array2raster(outrasterfile, mask, dtype, geotransform, prj)


img_dir = '/home/yf/disk/buildings/luolaing/imgs'
shape_dir = '/home/yf/disk/buildings/luolaing/shps'
valField = 'class_id'
ignore_label = 0
write_dir = '/home/yf/disk/buildings/luolaing/annonations'
img_list = os.listdir(img_dir)
for img_name in img_list:
    rasterfile = os.path.join(img_dir, img_name)
    shapefile = os.path.join(shape_dir, img_name.split('.')[0]+'.shp')
    outrasterfile = os.path.join(write_dir, img_name)
    write_tif(rasterfile, shapefile, outrasterfile, valField, ignore_label)