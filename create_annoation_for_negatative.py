import os
from PIL import Image
import json
import re
import random
import numpy as np

json_file=''
specific_imgpath=''
img_dir=''
json_dir=''
img_list=os.listdir(img_dir)
specific_img = np.array(Image.open(specific_imgpath))
w, h, c = specific_img.shape
data = json.load(json_file)
for img_id in img_list:
    img_path = os.path.join(img_dir, img_id)
    img = np.arrcy(Image.open(img_path))
    img[:w,:h,:] = specific_img
    tmp_json = os.path.join(img_dir, img_id.split('.')[0]+'.json')
    tmp = data.copy()
    tmp['imagePath'] = img_path
    tmp['imageHeight'],  tmp['imageHeight'], _ = img.shape
    with open(tmp_json, 'w') as outfile:
        json.dump(tmp, outfile)