from functools import partial
from multiprocessing.pool import Pool
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
from math import *
Image.MAX_IMAGE_PIXELS = None
import cv2
from numba import jit
# print('ddd')

def imread(image_path):
    img = cv2.imread(image_path)
    return img



# @jit
# def div(img, weight=1600, height=1600, stride=1000):
#     ori_w, ori_h = img.shape[0], img.shape[1]
#     # pad_w  = self.ceil_modulo(abs(ori_w-self.weight), self.stride) - ori_w
#     # pad_h = self.ceil_modulo(abs(ori_h-self.height), self.stride) - ori_h
#     pad_w = int(ceil(abs(ori_w-weight)/stride)*stride-ori_w + weight)
#     pad_h = int(ceil(abs(ori_h-height)/stride)*stride-ori_h + height)
#     # print(pad_w)
#     pad_img = np.pad(img, ((0, pad_w), (0, pad_h), (0, 0)),
#                     'constant', constant_values=255)
#     new_w, new_h = pad_img.shape[0], pad_img.shape[1]
#     num_w = int((new_w-weight)/stride)+1
#     num_h = int((new_h-height)/stride)+1
#     # count=0
#     # print(num_w)
#     for w_id in range(num_w):
#         for h_id in range(num_h):
#             tmp_img = pad_img[w_id*stride:w_id*stride +
#                             weight, h_id*stride:h_id*stride+height]
# @jit(nopython=False)
def divided(image_path, target_dir, weight=1600, height=1600, stride=1000):

    img = cv2.imread(image_path)
    # index = np.isnan(img)
    # img[index] = 255
    # values = np.unique(img)
    # print(values)
    # img[img==5] = 255
    # values = np.unique(img)
    # print(values)
    ori_w, ori_h = img.shape[0], img.shape[1]
    # pad_w  = self.ceil_modulo(abs(ori_w-self.weight), self.stride) - ori_w
    # pad_h = self.ceil_modulo(abs(ori_h-self.height), self.stride) - ori_h
    pad_w = int(ceil(abs(ori_w-weight)/stride)*stride-ori_w + weight)
    pad_h = int(ceil(abs(ori_h-height)/stride)*stride-ori_h + height)
    # print(pad_w)
    pad_img = np.pad(img, ((0, pad_w), (0, pad_h), (0, 0)),
                    'constant', constant_values=255)
    new_w, new_h = pad_img.shape[0], pad_img.shape[1]
    num_w = int((new_w-weight)/stride)+1
    num_h = int((new_h-height)/stride)+1
    # count=0
    # print(num_w)
    for w_id in range(num_w):
        for h_id in range(num_h):
            tmp_img = pad_img[w_id*stride:w_id*stride +
                            weight, h_id*stride:h_id*stride+height]
            # tmp_img = Image.fromarray(tmp_img)
            target_path = os.path.basename(image_path).split(
                '.')[0]+'_'+str(w_id)+'_'+str(h_id)+'.tif'
            target_path = os.path.join(target_dir, target_path)
            cv2.imwrite(target_path, tmp_img)


if __name__ == "__main__":
    # part = '2'
    # imgdir = '/home/yf/disk/object_detection/test/chai/' + part
    # target_dir = '/home/yf/disk/object_detection/test/divided/' + part
    imgdir = '/home/yf/disk/object_detection/test/tmp_test/test/'
    target_dir = '/home/yf/disk/object_detection/test/tmp_test/tmp/'
    partial_work = partial(divided, target_dir=target_dir)
    # pool = Pool(24)
    img_files = os.listdir(imgdir)
    image_paths = [os.path.join(imgdir, img_file) for img_file in img_files]
    with Pool(processes = 16) as pool:
        list = list(tqdm(pool.imap(partial_work, image_paths), total=len(img_files)))
    
