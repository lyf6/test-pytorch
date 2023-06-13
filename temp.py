import os
from shutil import copyfile
root='/home/yf/disk/buildings/Open_Cities_AI'
ls = ['0.1m', '0.5m', '0.7m']
sub_dir = os.listdir(root)
count = 0
destimg = os.path.join(root, 'img')
destmask = os.path.join(root, 'masks')
if not os.path.exists(destimg):
    os.makedirs(destimg)
if not os.path.exists(destmask):
    os.makedirs(destmask)

for sub_sub in sub_dir:
    if sub_sub in ls:
        ori_dir = os.path.join(root, sub_sub, 'train_tier_1')
        tmp_list = os.listdir(ori_dir)
        for sub_sub_sub in tmp_list:
            final = os.path.join(ori_dir, sub_sub_sub)
            if os.path.isdir(final):
                img = os.path.join(final, 'res-image.tif')
                mask = os.path.join(final, 'mask.tif')
                copyfile(img, os.path.join(destimg, str(count)+'.tif'))
                copyfile(mask, os.path.join(destmask, str(count) + '.tif'))
                count = count + 1