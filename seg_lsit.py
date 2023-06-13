import os
import random
import shutil
img_source='/home/yf/disk/GID/Large-scale Classification_5classes/image_RGB'
mask_source='/home/yf/disk/GID/Large-scale Classification_5classes/label_5classes'
trainnum=120
count=0

target_train_img='/home/yf/disk/gid/train/images'
target_train_label='/home/yf/disk/gid/train/masks'
target_test_img='/home/yf/disk/gid/test/images'
target_test_label='/home/yf/disk/gid/test/masks'
filelist=os.listdir(img_source)
random.shuffle(filelist)
for file in filelist:
    tmp_path_img=os.path.join(img_source,file)
    tmp_path_mask = os.path.join(mask_source, file.split('.t')[0]+'_label.tif')
    if count<trainnum:
        shutil.copy(tmp_path_img, target_train_img)
        shutil.copy(tmp_path_mask, target_train_label)
    else:
        shutil.copy(tmp_path_img, target_test_img)
        shutil.copy(tmp_path_mask, target_test_label)
    count+=1