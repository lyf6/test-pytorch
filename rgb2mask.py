import numpy as np
import os
from PIL import Image

def _rgb2mask(rgbmask, colormap):
    w, h, _=rgbmask.shape
    mask=np.ones(shape=(w,h),dtype=np.uint8)*(255)
    for id, rgb in enumerate(colormap):
        #index=rgbmask==rgb
        index=(rgbmask == rgb).all(-1)
        #index = zip(rgbmask ==rgb)
        mask[index]=id
    return mask.copy()

dir='/home/yf/disk/road/globe_road/train_rgb_mask'
target='/home/yf/disk/road/globe_road/train_mask'
files=os.listdir(dir)
# colormap=[[0, 200, 0],[150, 250, 0],
#           [150, 200, 150], [200, 0, 200],
#           [150, 0, 250], [150, 150, 250],
#           [250, 200, 0], [200, 200, 0],
#           [200, 0, 0], [250, 0, 150],
#           [200, 150, 150], [250, 150, 150],
#           [0, 0, 200], [0, 150, 200],
#           [0, 200, 250], [0, 0, 0]] #color map for sparse compete
#colormap=[[255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 255, 0], [0, 0, 255],[0, 0, 0]]  # color map for gid
#colormap = [[128,0,128], [128,128,0], [0, 0, 128], [128, 0, 0], [0,128,0]]
colormap = [[0, 0, 0], [255, 255, 255]] #color for globe road
for file in files:
    path=os.path.join(dir,file)
    img=np.array(Image.open(path))
    #print(img)
    mask=_rgb2mask(img, colormap)
    mask=Image.fromarray(mask)
    target_path=file.split('mask.p')[0]+'sat.png'
    target_path=os.path.join(target,target_path)
    mask.save(target_path)
