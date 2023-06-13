import cv2
from matplotlib.pyplot import gray
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
from math import *
from osgeo import gdal
Image.MAX_IMAGE_PIXELS = None
# print('ddd')
dir = '/home/yf/disk/object_detection/uavcar/paper_data/unlabeled_ori_images'
target_dir = '/home/yf/disk/object_detection/uavcar/paper_data/crops/'
files = os.listdir(dir)
weight = 640
height = 640
stride = 512

if not os.path.isdir(target_dir):
    os.makedirs(target_dir)


def divided(image_path, target_dir, weight=512, height=512, stride=256):

    # img = np.array(Image.open(image_path), dtype=np.uint8)
    img_handle = gdal.Open(image_path)
    img = np.array(img_handle.ReadAsArray(), dtype=np.uint8)
    if(img.ndim > 2):
        img = img.transpose((1, 2, 0))
    # img = img.transpose((1, 2, 0))
    #img = np.array(Image.open(image_path))
    # index = img == 0
    # img[index] = 255
    # img = img - 1
    # index = img == 254
    # img[index] = 255

    # index = np.isnan(img)
    # img[index] = 255
    # values = np.unique(img)
    # print(values)
    # img[img==5] = 255
    # values = np.unique(img)
    # print(values)
    ori_w, ori_h = img.shape[0], img.shape[1]
    # pad_w  = self.ceil_modulo(abs(ori_w-self.weight), self.stride) - ori_w
    # pad_h = self.ceil_modulo(abs(ori_h-self.height), self.stride) - ori_h
    pad_w = int(ceil(abs(ori_w-weight)/stride)*stride-ori_w + weight)
    pad_h = int(ceil(abs(ori_h-height)/stride)*stride-ori_h + height)
    # print(pad_w)
    if(img.ndim > 2):
        pad_img = np.pad(img, ((0, pad_w), (0, pad_h), (0, 0)),
                         'constant', constant_values = 255)
    else:
        pad_img = np.pad(img, ((0, pad_w), (0, pad_h)),
                         'constant', constant_values=255)
    new_w, new_h = pad_img.shape[0], pad_img.shape[1]
    num_w = int((new_w-weight)/stride)+1
    num_h = int((new_h-height)/stride)+1
    # count=0
    # print(num_w)
    for w_id in range(num_w):
        for h_id in range(num_h):
            tmp_img = pad_img[w_id*stride:w_id*stride +
                              weight, h_id*stride:h_id*stride+height]
            tmp_img = Image.fromarray(tmp_img)
            target_path = os.path.basename(image_path).split(
                '.')[0]+'_'+str(w_id)+'_'+str(h_id)+'.jpg'
            target_path = os.path.join(target_dir, target_path)
            tmp_img.save(target_path)


for img_name in files:
    if(img_name.endswith('.tif')):
        image_path = os.path.join(dir, img_name)
        divided(image_path, target_dir, weight=weight,
                height=height, stride=stride)
