import os
from PIL import Image
import numpy as np

dir = '/home/yf/disk/greenhouse/labels'
imgs = os.listdir(dir)
target_dir = dir
for img in imgs:
    new_name = os.path.join(target_dir, img.replace('_lab_', '_'))
    old_name = os.path.join(dir, img)
    os.rename(old_name, new_name)
