import numpy as np
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def _mask2rgb(mask, colormap):
    w,h =mask.shape
    img_mask=np.ones(shape=(w,h,3),dtype=np.uint8)*(0)
    for id, rgb in enumerate(colormap):
        #index=rgbmask==rgb
        index=(mask == id)
        #index = zip(rgbmask ==rgb)
        color=colormap[id]
        img_mask[index]=color
    return img_mask.copy()

# inter = 'gid_fcn'
# dir='/home/yf/Documents/mmsegmentation/work_dirs/'+inter+'/pred'
# target='/home/yf/disk/tmp/'+inter
target = '/home/yf/Documents/mmsegmentation/work_dirs/globeroad_ocrnet/results_rgb'
dir = '/home/yf/Documents/mmsegmentation/work_dirs/globeroad_ocrnet/results'
files=os.listdir(dir)
# colormap=[[0, 200, 0],[150, 250, 0],
#            [150, 200, 150], [200, 0, 200],
#            [150, 0, 250], [150, 150, 250],
#            [250, 200, 0], [200, 200, 0],
#            [200, 0, 0], [250, 0, 150],
#            [200, 150, 150], [250, 150, 150],
#            [0, 0, 200], [0, 150, 200],
#            [0, 200, 250], [0, 0, 0]] #color map for sparse compete
#colormap=[[255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 255, 0], [0, 0, 255],[0, 0, 0]]  # color map for gid
#colormap = [[255,0,0],[0,255,0],[0,0,0]]
# colormap=[[0, 200, 0],[150, 250, 0],
#            [150, 200, 150], [200, 0, 200],
#            [150, 0, 250], [150, 150, 250],
#            [250, 200, 0], [200, 200, 0],
#            [200, 0, 0], [250, 0, 150],[0, 150, 200]] #color map for tianchi
colormap = [[0,0,0],[255,255,255]] # globeroad
if not os.path.exists(target):
    os.makedirs(target)

for file in files:
    path=os.path.join(dir,file)
    img=np.array(Image.open(path))
    mask=_mask2rgb(img, colormap)
    mask=Image.fromarray(mask)
    target_path=file.split('.')[0]+'_mask.png'
    target_path=os.path.join(target,target_path)
    mask.save(target_path)