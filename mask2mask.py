import numpy as np
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# def _mask2mask(mask, colormap):
#     w,h =mask.shape
#     img_mask=np.ones(shape=(w,h,3),dtype=np.uint8)*(0)
#     for id, rgb in enumerate(colormap):
#         #index=rgbmask==rgb
#         index=(mask == id)
#         #index = zip(rgbmask ==rgb)
#         color=colormap[id]
#         img_mask[index]=color
#     return img_mask.copy()

# inter = 'gid_fcn'
# dir='/home/yf/Documents/mmsegmentation/work_dirs/'+inter+'/pred'
# target='/home/yf/disk/tmp/'+inter
target = '/home/yf/disk/whu/part2/satellite/masks'
#
dir = '/home/yf/disk/whu/part2/satellite/label'
# '/home/yf/Documents/py3.8/mmsegmentation/work_dirs/tianchi_deeplabv3plus/results'
#'/home/yf/Documents/py3.8/mmsegmentation/work_dirs/tianchi_deeplabv3plus/results'
files=os.listdir(dir)
#print(files)
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

if not os.path.exists(target):
    os.makedirs(target)

for file in files:
    path=os.path.join(dir,file)
    img=np.array(Image.open(path))
    #print(np.unique(img))
    img[img==255] = 1
    #img=img+1
    # mask=_mask2rgb(img, colormap)
    img=Image.fromarray(img)
    target_path=file.split('.tif')[0]+'.tif'
    target_path=os.path.join(target,target_path)
    print(target_path)
    img.save(target_path, quality=100, subsampling=0)