import os
import sys
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from math import *
from os.path import basename
import gdal
def main(img_dir, mask_dir, dest):

    dest_imgdir = os.path.join(dest,'img')
    dest_maskdir = os.path.join(dest,'mask')
    ignore_label = 255
    if not os.path.exists(dest_imgdir):
        os.makedirs(dest_imgdir)
    if not os.path.exists(dest_maskdir):
        os.makedirs(dest_maskdir)
    img_list = os.listdir(mask_dir)
    for img_name in img_list:
        mask_path = os.path.join(mask_dir,img_name)
        raster_basename = img_name.split('.')[0]
        img_path = os.path.join(img_dir,raster_basename+'.tif')

        tif = gdal.Open(img_path)
        
        img = np.array(tif.ReadAsArray(),dtype=np.uint8)
        if img.ndim > 2:
            img = img.transpose((1,2,0))
        

        #img = np.array(Image.open(img_path))[:,:,:3]

        mask = np.array(Image.open(mask_path),dtype=np.uint8)
        print(mask.shape)
        mask[mask>0] = 1
        # values = np.unique(mask)
        # print(values)
        stride = 512
        crop_size = 512
        ori_w, ori_h = mask.shape
        pad_w = (ceil(ori_w/stride))*stride-ori_w
        pad_h = (ceil(ori_h/stride))*stride-ori_h
        pad_raster = np.pad(img,((0, pad_w), (0, pad_h),(0,0)), 'constant', constant_values=0)
        pad_roi = np.pad(mask,((0, pad_w), (0, pad_h)), 'constant', constant_values=255)
        new_w, new_h = pad_roi.shape
        pad_roi[np.isnan(pad_roi)] = ignore_label
        pad_raster[np.isnan(pad_raster)] = 0
        new_w = int(new_w/crop_size)
        new_h = int(new_h/crop_size)
        for w_id in range(new_w):
            for h_id in range(new_h):
                tmp_roi = pad_roi[w_id*stride:w_id*stride+crop_size, h_id*stride:h_id*stride+crop_size]
                tmp_img = pad_raster[w_id*stride:w_id*stride+crop_size, h_id*stride:h_id*stride+crop_size,:]
                tmp_roi_path = os.path.join(dest_maskdir, raster_basename+'_'+str(w_id)+'_'+str(h_id)+'.tif')
                tmp_img_path = os.path.join(dest_imgdir, raster_basename+'_'+str(w_id)+'_'+str(h_id)+'.tif')
                print(tmp_img_path)
                tmp_roi = Image.fromarray(tmp_roi)
                tmp_img = Image.fromarray(tmp_img)
                tmp_roi.save(tmp_roi_path,quality=100, subsampling=0)
                tmp_img.save(tmp_img_path,quality=100, subsampling=0)

base_dir = '/home/yf/disk/buildings/extract_regions/'
img_dir = os.path.join(base_dir,'images')
mask_dir = os.path.join(base_dir,'masks')
dest = os.path.join(base_dir,'created')
main(img_dir, mask_dir, dest)