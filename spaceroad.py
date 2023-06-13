import os
from PIL import Image
import numpy as np

source='/home/yf/disk/spaceroad/val/gt'
target='/home/yf/disk/spaceroad/val/mask'
lists=os.listdir(source)
for img_id in lists:
    img_path=os.path.join(source,img_id)
    img=np.array(Image.open(img_path))
    index=img>=194
    img[:]=0
    img[index]=255
    img=Image.fromarray(img)
    target_path=os.path.join(target,img_id)
    img.save(target_path)
