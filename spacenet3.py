from PIL import Image
import os.path as osp
import os
import numpy as np

source4 = '/home/yf/disk/road/spacenet3/labels/'
source12 = '/home/yf/disk/spacenet3/masks/'
target = '/home/yf/disk/spacenet3/buffmask/'
imgs_ls = os.listdir(source4)
for img_name in imgs_ls:
    img_path4 = osp.join(source4, img_name)
    img_path12 = osp.join(source12, img_name)
    img4 = np.array(Image.open(img_path4))
    img12 = np.array(Image.open(img_path12))
    index = img4==1
    img12[index] = 1
    save_path = osp.join(target, img_name)
    img = Image.fromarray(img12)
    img.save(save_path)