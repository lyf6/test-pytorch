from functools import partial
from multiprocessing.pool import Pool
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
from math import *
Image.MAX_IMAGE_PIXELS = None
import cv2
# print('ddd')
import torch.distributed as dist
import torch.utils.data.distributed
import torch
from torch.utils.data import Dataset
import os
import cv2
import argparse
parser = argparse.ArgumentParser()
from tqdm import tqdm
# 注意这个参数，必须要以这种形式指定，即使代码中不使用。因为 launch 工具默认传递该参数
parser.add_argument("--local_rank", type=int)



class data(Dataset):
    def __init__(self, imgdir, target_dir, weight=1600, height=1600, stride=1000):
        self.img_list = os.listdir(imgdir)
        self.imgdir = imgdir
        self.target_dir = target_dir
        # if not os.path.exists(target_dir):
        #     os.makedirs(self.target_dir)
        self.weight = weight
        self.height = height
        self.stride = stride
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        image_name = self.img_list[idx]
        image_path = os.path.join(self.imgdir, image_name)
        self.divided(image_path)
        return image_name
    
    def ceil_modulo(self, x, mod):
        if x % mod == 0:
            return x
        return (x // mod + 1) * mod

    def divided(self, image_path):

        img = cv2.imread(image_path)
        # index = np.isnan(img)
        # img[index] = 255
        # values = np.unique(img)
        # print(values)
        #img[img==5] = 255
        #values = np.unique(img)
        # print(values)
        ori_w, ori_h = img.shape[0], img.shape[1]
        # pad_w  = self.ceil_modulo(abs(ori_w-self.weight), self.stride) - ori_w
        # pad_h = self.ceil_modulo(abs(ori_h-self.height), self.stride) - ori_h
        pad_w = int(ceil(abs(ori_w-self.weight)/self.stride)*self.stride-ori_w + self.weight)
        pad_h = int(ceil(abs(ori_h-self.height)/self.stride)*self.stride-ori_h + self.height)
        # print(pad_w)
        pad_img = np.pad(img, ((0, pad_w), (0, pad_h), (0, 0)),
                        'constant', constant_values=255)
        new_w, new_h = pad_img.shape[0], pad_img.shape[1]
        num_w = int((new_w-self.weight)/self.stride)+1
        num_h = int((new_h-self.height)/self.stride)+1
        # count=0
        # print(num_w)
        for w_id in range(num_w):
            for h_id in range(num_h):
                tmp_img = pad_img[w_id*self.stride:w_id*self.stride +
                                self.weight, h_id*self.stride:h_id*self.stride+self.height]
                # tmp_img = Image.fromarray(tmp_img)
                target_path = os.path.basename(image_path).split(
                    '.')[0]+'_'+str(w_id)+'_'+str(h_id)+'.jpg'
                target_path = os.path.join(self.target_dir, target_path)
                cv2.imwrite(target_path, tmp_img)



if __name__ == "__main__":
    # target_dir = '/home/yf/disk/object_detection/test/divided'
    # partial_work = partial(divided, target_dir=target_dir)
    # # pool = Pool(24)
    # dir = '/home/yf/disk/object_detection/test/ori_images/'
    # img_files = os.listdir(dir)
    # image_paths = [os.path.join(dir, img_file) for img_file in img_files]
    # with Pool(processes = 24) as pool:
    #     list = list(tqdm(pool.imap(partial_work, image_paths), total=len(img_files)))
    # rank = int(os.environ["RANK"])
    # local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    args = parser.parse_args()
    imgdir = '/home/yf/disk/object_detection/test/tmp_test/test'
    target_dir = '/home/yf/disk/object_detection/test/divided'
    dataset = data(imgdir, target_dir)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        shuffle=False,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        sampler=train_sampler,
    )
    for idx, targets in enumerate(tqdm(train_loader)): 
        print(targets)