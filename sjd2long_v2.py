import re
import os
import gdal
import csv
from PIL import Image
import numpy as np

file = '/home/yf/Documents/test-pytorch/2019jh-crop.csv'
imgdir = '/home/yf/disk/myannotated_cd/test'
fp = open(file)
lines = fp.read().splitlines()
target_csv = '2019jh_long-crop.csv'
myFile = open(target_csv,'w',encoding='utf8',newline='')
writer = csv.writer(myFile)

for line in lines:

    #imgname = line.split(' ')[0].split('\\')[1].split('.png')[0]+'.tif'
    imgname = (line.split(' ')[0].split('.png')[0]+'.tif').split('"')[1]
    #print(imgname)
    label_name = line.split(' ')[1]
    pixel_loc = re.findall(r'[(](.*?)[)]', line)
    imgpath = os.path.join(imgdir, imgname)
    tif = gdal.Open(imgpath)
    gt = tif.GetGeoTransform()
    print(imgpath)


    x_min = gt[0]-39000000  #qu chu daihao yingxiang
    #x_min = gt[0]
    x_size = gt[1]
    y_min = gt[3]
    y_size = gt[5]
    pixel_list = []
    pixel_list.append(label_name)
    for loc in pixel_loc:
        tmp = loc.split(',')
        p_x = float(tmp[0])
        p_y = float(tmp[1])
        l_x = p_x*x_size + x_min
        l_y = p_y*y_size + y_min
        pixel_list.append(l_x)
        pixel_list.append(l_y)
    writer.writerow(pixel_list)
