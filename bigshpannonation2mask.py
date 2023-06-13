# -*- coding: UTF-8 -*-
# import ogr
from osgeo import osr, gdal
import numpy as np
import sys
import os
from shapely.geometry import Polygon
import geopandas as gpd
from PIL import Image, ImageDraw
# from combine import array2raster
import pandas as pd


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


def multi2single(gpdf):
    gpdf_singlepoly = gpdf[gpdf.geometry.type == 'Polygon']
    gpdf_multipoly = gpdf[gpdf.geometry.type == 'MultiPolygon']

    for i, row in gpdf_multipoly.iterrows():
        Series_geometries = pd.Series(row.geometry)
        df = pd.concat([gpd.GeoDataFrame(row, crs=gpdf_multipoly.crs).T]
                       * len(Series_geometries), ignore_index=True)
        df['geometry'] = Series_geometries
        gpdf_singlepoly = pd.concat([gpdf_singlepoly, df])

    gpdf_singlepoly.reset_index(inplace=True, drop=True)
    return gpdf_singlepoly


def shape2mask(raster, shape, ignore_label):
    dataset = gdal.Open(raster)
    col = dataset.RasterXSize
    row = dataset.RasterYSize
    geotransform = dataset.GetGeoTransform()
    prj = dataset.GetProjection()
    distY, distX = abs(geotransform[5]), abs(geotransform[1])
    ulY, ulX = geotransform[3], geotransform[0]  # startx starty
    endY = ulY - row*distY
    endX = ulX + col*distX
    image_region = gpd.GeoDataFrame({'geometry': gpd.GeoSeries(
        Polygon([(ulX, ulY), (endX, ulY), (endX, endY), (ulX, endY)]))})

    layer = gpd.read_file(shape)
    layer = multi2single(layer)
    rasterPoly = Image.new("I", (col, row), ignore_label)
    rasterize = ImageDraw.Draw(rasterPoly)
    # for geo in geometry:
    #     geo['area'] = geo.area
    layer['area'] = layer.apply(lambda row: row.geometry.area, axis=1)
    layer.sort_values(by='area', ascending=False, inplace=True)
    insection_polygon = gpd.overlay(image_region, layer, how='intersection')

    geometry = insection_polygon.geometry
    # class_id = layer[valField]
    class_label = 1
    for geo in geometry:
        # print(geo)
        # if class_label != 0:
        if(geo.geom_type == 'Polygon'):
            pixels = []
            xs, ys = geo.exterior.coords.xy
            for x, y, in zip(xs, ys):
                pixel_x = abs(int((x - ulX) / distX))
                pixel_y = abs(int((y - ulY) / distY))
                pixel_y = row if pixel_y > row else pixel_y
                pixel_x = col if pixel_x > col else pixel_x
                pixels.append((pixel_x, pixel_y))
            # print(pixels)
            rasterize.polygon(pixels, fill=class_label)
    mask = np.array(rasterPoly, dtype=np.uint8)
    return mask, geotransform, prj


def main():

    # keys = ['bd', 'jh', 'jx']
    
    shp_dir = '/home/yf/disk/jinzhongqiao/shp/'
    rasterfile = '/home/yf/disk/jinzhongqiao/Production_3_ortho_merge.tif'
   
    outrasterfile = '/home/yf/disk/jinzhongqiao/ann/Production_3_ortho_merge.tif'

    shaplefiles = [fs for fs in os.listdir(shp_dir) if fs.endswith('.shp')]
    ignore_label = 0
    
    for shaple_name in shaplefiles:
        shaplefile = os.path.join(shp_dir, shaple_name)
        mask, geotransform, prj = shape2mask(
                rasterfile, shaplefile, ignore_label)
        dtype = "Byte"
        array2raster(outrasterfile, mask,
                            dtype, geotransform, prj)


if __name__ == '__main__':
    main()
