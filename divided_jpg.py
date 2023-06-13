import os
from PIL import Image
from math import *
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import os

dir = '/home/yf/disk/object_detection/uavcar/paper_data/unlabeled_ori_images'
target_dir = '/home/yf/disk/object_detection/uavcar/paper_data/unlabeled/'
files = os.listdir(dir)
weight = 640
height = 640
stride = 512

if not os.path.isdir(target_dir):
    os.makedirs(target_dir)

def divided(image_path, target_dir, weight=512, height=512, stride=256):
    

    #print('files')
    weight=weight
    height=height
    stride=stride

    img=np.array(Image.open(image_path))
    # index = np.isnan(img)
    # img[index] = 0
    # values = np.unique(img)
    #print(values)
    #img[img==5] = 255
    #values = np.unique(img)
    #print(values)
    ori_w, ori_h= img.shape[0], img.shape[1]
    pad_w = int((ceil(ori_w/stride))*stride-ori_w)
    pad_h = int((ceil(ori_h/stride))*stride-ori_h)
    #print(pad_w)
    pad_img=np.pad(img,((0,pad_w),(0,pad_h),(0,0)),'constant', constant_values=0)
    new_w, new_h = pad_img.shape[0], pad_img.shape[1]
    num_w = int(new_w/weight)
    num_h = int(new_h/height)
    #count=0
    #print(num_w)
    for w_id in range(num_w):
        for h_id in range(num_h):
            tmp_img=pad_img[w_id*stride:w_id*stride+weight, h_id*stride:h_id*stride+height]
            tmp_img=Image.fromarray(tmp_img)
            target_path=os.path.basename(image_path).split('.')[0]+'_'+str(w_id)+'_'+str(h_id)+'.jpg'
            target_path=os.path.join(target_dir,target_path)
            tmp_img.save(target_path)

for img_name in files:
    if(img_name.endswith('.jpg')):
        image_path = os.path.join(dir, img_name)
        divided(image_path, target_dir, weight=weight,
                height=height, stride=stride)