import numpy as np
import os
from PIL import Image
from math import *
Image.MAX_IMAGE_PIXELS = None


imgpath="/home/yf/disk/feilianghua/rice_test/202206yue_0_.tif"
target_dir="/home/yf/disk/feilianghua/rice_divided/"

weight=224
height=224
stride=224

img=np.array(Image.open(imgpath))
ori_w, ori_h, _ = img.shape
pad_w = (ceil(ori_w/stride))*stride-ori_w
pad_h = (ceil(ori_h/stride))*stride-ori_h
pad_img=np.pad(img,((0,pad_w),(0,pad_h),(0,0)),'constant', constant_values=0)
new_w, new_h,_= pad_img.shape
num_w = int(new_w/weight)
num_h = int(new_h/height)
count=0
for w_id in range(num_w):
    for h_id in range(num_h):
        
        tmp_img=pad_img[w_id*stride:w_id*stride+weight, h_id*stride:h_id*stride+height,:]
        tmp_img=Image.fromarray(tmp_img)
        target_path=str(count)+'.jpg'
        print(target_path)
        target_path=os.path.join(target_dir,target_path)
        tmp_img.save(target_path)
        count=count+1