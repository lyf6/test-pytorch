import os
from PIL import Image
import numpy as np

dir = '/home/yf/disk/camera/test_tmp'
imgs = os.listdir(dir)
target_dir = '/home/yf/disk/camera/test'
for img in imgs:
    name = img.strip().split()[2].split('.')[0]+'.tif'
    #name = img.strip().split()[2]
    path = os.path.join(dir, img)
    tmp = Image.fromarray(np.array(Image.open(path))[300:, :1200])
    #tmp = Image.fromarray(np.array(Image.open(path))[300:, :1200])
    target_path = os.path.join(target_dir, name)
    tmp.save(target_path)