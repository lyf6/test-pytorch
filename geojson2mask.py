import geopandas as gpd
# import rasterio
# import fiona
# from rasterio import features
# import os
# from datetime import datetime as dt
from PIL import Image, ImageDraw
import pandas as pd
import gdal
import numpy as np
import os

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

def shape2mask(raster, geojson, dest_path, back_label=0):
    dataset = gdal.Open(raster)
    col = dataset.RasterXSize
    row = dataset.RasterYSize
    geotransform = dataset.GetGeoTransform()
    prj = dataset.GetProjection()
    ulY, ulX = geotransform[3], geotransform[0]
    distY, distX = abs(geotransform[5]), abs(geotransform[1])
    try:
        layer = gpd.read_file(geojson)
        layer = multi2single(layer)
        layer['area'] = layer.apply(lambda row: row.geometry.area, axis=1)
        layer.sort_values(by='area', ascending=False, inplace=True)
        geometry = layer.geometry
        

    except:
        geometry = []
    # layer = gpd.read_file(shape)
    # layer = multi2single(layer)
    rasterPoly = Image.new("I", (col, row), back_label)
    rasterize = ImageDraw.Draw(rasterPoly)

    # for geo in geometry:
    #     geo['area'] = geo.area
    # layer['area'] = layer.apply(lambda row: row.geometry.area, axis=1)
    # layer.sort_values(by='area', ascending=False, inplace=True)
    # geometry = layer.geometry
    # class_id = layer[valField]
    for geo in geometry:
        # print(geo)
        #if class_label != 0:
        pixels = []
        xs, ys = geo.exterior.coords.xy

        for x, y,  in zip(xs, ys):
            pixel_x = abs(int((x - ulX) / distX))
            pixel_y = abs(int((y - ulY) / distY))
            pixel_y = row if pixel_y > row else pixel_y
            pixel_x = col if pixel_x > col else pixel_x
            pixels.append((pixel_x, pixel_y))
        #print(pixels)
        rasterize.polygon(pixels, fill=1)
    mask = np.array(rasterPoly,np.uint8)
    mask = Image.fromarray(mask)
    mask.save(dest_path,quality=100, subsampling=0)
def masks_from_geojsons(geojson_dir, im_src_dir, mask_dest_dir,
                        skip_existing=False, verbose=False):
    """Create mask images from geojsons.

    Arguments:
    ----------
    geojson_dir (str): Path to the directory containing geojsons.
    im_src_dir (str): Path to a directory containing geotiffs corresponding to
        each geojson. Because the georegistration information is identical
        across collects taken at different nadir angles, this can point to
        geotiffs from any collect, as long as one is present for each geojson.
    mask_dest_dir (str): Path to the destination directory.

    Creates a set of binary image tiff masks corresponding to each geojson
    within `mask_dest_dir`, required for creating the training dataset.

    """
    if not os.path.exists(geojson_dir):
        raise NotADirectoryError(
            "The directory {} does not exist".format(geojson_dir))
    if not os.path.exists(im_src_dir):
        raise NotADirectoryError(
            "The directory {} does not exist".format(im_src_dir))
    geojsons = [f for f in os.listdir(geojson_dir) if f.endswith('json')]
    ims = [f for f in os.listdir(im_src_dir) if f.endswith('.tif')]
    #print(ims)
    for geojson in geojsons:
        tmp = geojson.split('_')
        chip_id = tmp[-2]+'_' + tmp[-1]
        chip_id = chip_id.split('.')[0]
        #print(chip_id) 

        
        matching_im_list = [i for i in ims if chip_id in i]
        if len(matching_im_list) > 0:
            matching_im = matching_im_list[0]
            dest_path = os.path.join(mask_dest_dir, matching_im.split('.')[0]+'.tif')
            raster = os.path.join(im_src_dir, matching_im)
            print(dest_path)
            geo = os.path.join(geojson_dir, geojson)
            shape2mask(raster,geo, dest_path)



    #return mask

train_src_dir='/home/yf/disk/buildings/spacenet4'
train_mask_dir='/home/yf/disk/buildings/spacenet4/masks'
geojson_src_dir = os.path.join(train_src_dir, 'geojson_buildings')
if not os.path.exists(train_mask_dir):
    os.makedirs(train_mask_dir)
COLLECTS = ['nadir7_catid_1030010003D22F00',
            'nadir8_catid_10300100023BC100',
            'nadir10_catid_1030010003CAF100',
            'nadir10_catid_1030010003993E00',
            'nadir13_catid_1030010002B7D800',
            'nadir14_catid_10300100039AB000',
            'nadir16_catid_1030010002649200',
            'nadir19_catid_1030010003C92000',
            'nadir21_catid_1030010003127500',
            'nadir23_catid_103001000352C200',
            'nadir25_catid_103001000307D800',
            'nadir27_catid_1030010003472200',
            'nadir29_catid_1030010003315300',
            'nadir30_catid_10300100036D5200',
            'nadir32_catid_103001000392F600',
            'nadir34_catid_1030010003697400',
            'nadir36_catid_1030010003895500',
            'nadir39_catid_1030010003832800',
            'nadir42_catid_10300100035D1B00',
            'nadir44_catid_1030010003CCD700',
            'nadir46_catid_1030010003713C00',
            'nadir47_catid_10300100033C5200',
            'nadir49_catid_1030010003492700',
            'nadir50_catid_10300100039E6200',
            'nadir52_catid_1030010003BDDC00',
            'nadir53_catid_1030010003CD4300',
            'nadir53_catid_1030010003193D00']
for collect in COLLECTS:
    masks_from_geojsons(geojson_src_dir,
                        os.path.join(train_src_dir,collect,'PS-RGBNIR'),
                        train_mask_dir)





# tif = 'G://buildings//spacenet4//nadir7_catid_1030010003D22F00//PS-RGBNIR//SN4_buildings_train_AOI_6_Atlanta_nadir7_catid_1030010003D22F00_PS-RGBNIR_732701_3730989.tif'
# geojson='G://buildings//spacenet4//geojson_buildings//SN4_buildings_train_AOI_6_Atlanta_geojson_buildings_732701_3730989.geojson'
# shape2mask(tif, geojson)
