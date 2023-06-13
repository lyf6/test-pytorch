import numpy as np
import os
from PIL import Image
from math import *
Image.MAX_IMAGE_PIXELS = None
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=520)
# path = '/home/yf/disk/gid/div_train/masks/GF2_PMS1__L1A0000564539-MSS1_1.tif'
# img=np.array(Image.open(path))
# print(img)
source_dir = '/home/yf/disk/whu/part1/crop_imgs/val/label'
target_dir = '/home/yf/disk/whu/part1/crop_imgs/val/masks'
img_ls = os.listdir(source_dir)
count={id:0.0 for id in range(2)}
for img_id in img_ls:
    img_pt = os.path.join(source_dir, img_id)
    target_path = os.path.join(target_dir, img_id)
    img = np.array(Image.open(img_pt))
    for labeled in count:
        sum=(img==labeled).sum()
        count[labeled]+=sum
print(count)
    #print(img_pt)
    #print(np.array(Image.open(img_pt)).astype(np.uint8))
    #img=Image.fromarray(np.array(Image.open(img_pt)).astype(np.uint8))
    #img.save(target_path)