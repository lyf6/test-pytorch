import numpy as np
import os
from PIL import Image
from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = None
import cv2
from functools import partial
from multiprocessing.pool import Pool
from PIL.TiffTags import TAGS
import tifffile
#dir='/home/yf/disk/data/ucm'
    #'/home/yf/Documents/data/rss/div_test/ori_images'
#target_dir='/home/yf/Documents/data/rss/div_test/rgb_images'
#files=os.listdir(dir)

def tif2jpg(name, imgdir, jpg_dir):
    #print(os.path.join(root, name))
    #print(name)
    path=os.path.join(yourpath,name)
    img = tifffile.imread(path)
    print('ddd')
    # img_exif = None
    # if 'exif' in img.info:
    #     img_exif = img.info['exif']
    # # w, h = img.size
    # save_name = name.split('.tif')[0]+'.jpg'
    # meta_dict = {TAGS[key] : img.tag[key] for key in img.tag_v2}
    # print(meta_dict)
    # if img_exif is not None:
    #     img.save(os.path.join(jpg_dir, save_name), exif=img_exif, quality=100)
    # else:
    #     img.save(os.path.join(jpg_dir, save_name), quality=100, info=img.info, tag=img.tag, tag_v2 = img.tag_v2)
    # img.save(os.path.join(jpg_dir, save_name))
    # cv2.imwrite(os.path.join(jpg_dir, save_name), img)
    #print(path)

if __name__ == "__main__":
    part = '1'
    yourpath = '/home/yf/disk/object_detection/test/chai/tmp/' + part
    jpg_dir = '/home/yf/disk/object_detection/test/chai/jpg' + part
    partial_work = partial(tif2jpg,imgdir=yourpath, jpg_dir=jpg_dir)
    files = os.listdir(yourpath)
    if not os.path.exists(jpg_dir):
        os.makedirs(jpg_dir)
    with Pool(processes = 2) as pool:
        list = list(tqdm(pool.imap(partial_work, files), total=len(files)))
