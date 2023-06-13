import re
import os
import gdal
import csv

file = '/home/yf/Documents/test-pytorch/PtResult_test_results2.txt'
imgdir = '/home/yf/disk/testdata/2019jh'
fp = open(file)
lines = fp.read().splitlines()
target_csv = 'sjd.csv'
myFile = open(target_csv,'w',encoding='utf8',newline='')
writer = csv.writer(myFile)

for line in lines:
    if 'jh2019' in line:
        #imgname = line.split(' ')[0].split('\\')[1].split('.png')[0]+'.tif'
        imgname = line.split(' ')[0].split('.png')[0]+'.tif'
        print(imgname)
        pixel_loc = re.findall(r'[(](.*?)[)]', line)
        imgpath = os.path.join(imgdir, imgname)
        tif = gdal.Open(imgpath)
        gt = tif.GetGeoTransform()
        print(gt)
        x_min = gt[0]-39000000  #qu chu daihao yingxiang
        x_size = gt[1]
        y_min = gt[3]
        y_size = gt[5]
        pixel_list = []
        for loc in pixel_loc:
            tmp = loc.split(',')
            p_x = float(tmp[0])
            p_y = float(tmp[1])
            l_x = p_x*x_size + x_min
            l_y = p_y*y_size + y_min
            pixel_list.append(l_x)
            pixel_list.append(l_y)
        writer.writerow(pixel_list)
