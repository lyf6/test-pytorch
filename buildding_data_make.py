import os

import random

import shutil

img_source = '/home/yf/disk/spacenet2/8bit_ps'

mask_source = '/home/yf/disk/spacenet2/graymasks'

trainnum = 0.6

target_train_img = '/home/yf/disk/spacenet2/SV_train/images'

target_train_label = '/home/yf/disk/spacenet2/SV_train/masks'

target_test_img = '/home/yf/disk/spacenet2/SV_test/images'

target_test_label = '/home/yf/disk/spacenet2/SV_test/masks'

filelist = os.listdir(img_source)

random.shuffle(filelist)

#trainnum = int(len(filelist)*trainnum)

wanted_cls = ['Shanghai','Vegas']

if not os.path.exists(target_train_img):
    os.makedirs(target_train_img)

if not os.path.exists(target_train_label):
    os.makedirs(target_train_label)

if not os.path.exists(target_test_img):
    os.makedirs(target_test_img)

if not os.path.exists(target_test_label):
    os.makedirs(target_test_label)


for cls in wanted_cls:

    pre_cls = [filenames for filenames in filelist if cls in filenames]
    print(cls)
    wanted_count = int(trainnum*len(pre_cls))
    print(wanted_count, len(pre_cls))
    count = 0

    for tmpfile in pre_cls:

        tmp_path_img = os.path.join(img_source, tmpfile)

        tmp_path_mask = os.path.join(mask_source, tmpfile)

        if count < wanted_count:

            shutil.copy(tmp_path_img, target_train_img)

            shutil.copy(tmp_path_mask, target_train_label)

        else:

            shutil.copy(tmp_path_img, target_test_img)

            shutil.copy(tmp_path_mask, target_test_label)

        count += 1


