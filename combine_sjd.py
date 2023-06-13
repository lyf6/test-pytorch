from __future__ import division
import numpy as np
import os
from PIL import Image
from math import *
Image.MAX_IMAGE_PIXELS = None


dir='/home/yf/disk/SJD/ori_test/masks'
target_dir='/home/yf/disk/SJD/prod_comb'
result_path= '/home/yf/Documents/HRNet-Semantic-Segmentation-pytorch-v1.1/test_val_results'
files=os.listdir(dir)
weight=512
height=512
stride=512

for file in files:
    path=os.path.join(dir,file)
    img=np.array(Image.open(path))
    ori_w, ori_h = img.shape
    pad_w = int(ceil(ori_w/stride)*stride-ori_w)
    pad_h = int(ceil(ori_h/stride)*stride-ori_h)
    print(pad_w, pad_h)
    #print(ceil(ori_w/stride), ceil(ori_h/stride))
    #print(ori_w/stride, ori_h/stride)
    pad_img=np.pad(img,((0,pad_w),(0,pad_h)),'constant', constant_values=255) # all ignored labeld default 255
    new_w, new_h= pad_img.shape
    num_w = int(new_w/weight)-1
    num_h = int(new_h/height)-1
    count=0
    for w_id in range(num_w):
        for h_id in range(num_h):
            tmp_path = file.split('.tif')[0]  + '_' +str(count) + '.png'
            print(tmp_path)
            tmp_path = os.path.join(result_path,tmp_path)
            tmp_img = Image.open(tmp_path)
            pad_img[w_id * stride:w_id * stride + weight, h_id * stride:h_id * stride + height] = tmp_img
            count=count+1
    target_path = file.split('.ti')[0] +  '_pred.png'
    target_path = os.path.join(target_dir,target_path)
    pad_img = Image.fromarray(pad_img)
    pad_img.save(target_path)
