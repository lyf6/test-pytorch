import os
from tifffile import imsave
from tifffile import imread
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = None

dir='/home/yf/disk/qgis34-files/object'
files=os.listdir(dir)
target_dir='/home/yf/disk/tmp'
count={1:255}
for file in files:
    path = os.path.join(dir, file)
    img = np.array(Image.open(path))
    index = np.isnan(img)
    img[index] = 0
    for labeled in count.keys():
        img[img==labeled]=count[labeled]

    save_path=os.path.join(target_dir, file)
    #imsave(save_path, img)
    img = Image.fromarray(img)
    img.save(save_path)
