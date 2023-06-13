import os
from tifffile import imsave
from tifffile import imread
from PIL import Image
import numpy as np
import gdal, ogr, os, osr
Image.MAX_IMAGE_PIXELS = None

path='/home/yf/disk/myannotated_cd/2017-2019val/2017-testv2map.tif'
file='2017-testv2remap.tif'
target_dir='/home/yf/disk/myannotated_cd/2017-2019val'
count={4:1, 1:3, 2:2, 3:4, 5:4, 6:4}
#count={1:1, 2:3, 3:2, 4:4, 5:4, 6:4}
#count={0:1}
#path = os.path.join(target_dir, file)

tif = gdal.Open(path)
img = np.array(tif.ReadAsArray())
# print(img)
# index = np.isnan(img)
# img[index] = 255

tmp =np.ones_like(img)*255
#index = np.isnan(img)
# img[index] = 255
print(tmp)
for labeled in count.keys():
    tmp[img==labeled]=count[labeled]

save_path=os.path.join(target_dir, file)
#imsave(save_path, img)
print(tmp)
tmp = Image.fromarray(tmp)
tmp.save(save_path)