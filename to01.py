import numpy as np
import cv2
import os
from PIL import Image
from PIL import ImageFilter
Image.MAX_IMAGE_PIXELS = None

source_dir = '/home/yf/disk/output/sjd_v2/seg_hrnet_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484x2/test_results'
target_dir = '/home/yf/disk/output/sjd_v2/seg_hrnet_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484x2/test_results_01'

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

imglist = os.listdir(source_dir)
#print(imglist)
for img_path in imglist:
    path = os.path.join(source_dir, img_path)
    #print(source_dir,path)
    img = Image.open(path)
    #img = imgg.filter(ImageFilter.BLUR)
    #img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

    img = np.array(img)
    index_2 = (img==2)
    print(index_2)
    img[index_2] = 1
    img = Image.fromarray(img)
    target_path=os.path.join(target_dir,img_path)
    img.save(target_path)