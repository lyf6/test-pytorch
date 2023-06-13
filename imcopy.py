import os
from PIL import Image
import numpy as np
import shutil
import random

images_dir = '/home/yf/disk/greenhouse/images'
labels_dir = '/home/yf/disk/greenhouse/labels'
train_img_dir = '/home/yf/disk/greenhouse/exp_train_img'
train_label_dir = '/home/yf/disk/greenhouse/exp_train_label'
val_img_dir = '/home/yf/disk/greenhouse/exp_val_img'
val_label_dir = '/home/yf/disk/greenhouse/exp_val_label'
images = os.listdir(images_dir)
random.shuffle(images)
ratio=0.5
train_num = int(len(images)*0.5)
for i in range(len(images)):
    img = os.path.join(images_dir, images[i])
    label = os.path.join(labels_dir, images[i])
    if i<train_num:
        train_img = os.path.join(train_img_dir, images[i])
        train_label = os.path.join(train_label_dir, images[i])
        shutil.copyfile(img,train_img)
        shutil.copyfile(label,train_label)
    else:
        val_img = os.path.join(val_img_dir, images[i])
        val_label = os.path.join(val_label_dir, images[i])
        shutil.copyfile(img,val_img)
        shutil.copyfile(label,val_label)