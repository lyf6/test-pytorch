import os
from PIL import Image
import numpy as np


mask_dir = '/home/yf/disk/output/myann/seg_hrnet_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484x2/test_results'
img_dir = '/home/yf/disk/myannotated_cd/test'
save_dir = '/home/yf/disk/myannotated_cd/test_masks'
mask_list = os.listdir(mask_dir)
colormap = [[0,0,0],[128,0,128],[128,128,0],[0,0,128],[128,0,0],[0,128,0]]

for mask_path in mask_list:
    print(mask_path)
    target_path = os.path.join(save_dir, mask_path)
    img_path = os.path.join(img_dir, mask_path.split('.')[0]+'.tif')
    mask_path = os.path.join(mask_dir, mask_path)
    mask = np.array(Image.open(mask_path))
    img = np.array(Image.open(img_path))
    for id in range(1,6):
        index=(mask == id)
        color=colormap[id]
        img[index]=color
    img=Image.fromarray(img)

    img.save(target_path)