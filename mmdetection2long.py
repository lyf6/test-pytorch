import re
import os
import gdal
import csv

file = '/home/yf/disk/jh2019.csv'
imgdir = '/home/yf/disk/testdata/2019jh'
fp = open(file)
lines = fp.read().splitlines()
target_csv = 'jh2019_long.csv'
myFile = open(target_csv,'w',encoding='utf8',newline='')
writer = csv.writer(myFile)

for line in lines:
    if 'jh2019' in line:
        #imgname = line.split(' ')[0].split('\\')[1].split('.png')[0]+'.tif'
        #imgname = (line.split(' ')[0].split('.png')[0]+'.tif').split('"')[1]
        #print(imgname)

        imgname = line.split(',')[0]
        pixel_loc = line.split(',')[1:5]
        #re.findall(r'[(](.*?)[)]', line)
        imgpath = os.path.join(imgdir, imgname)
        tif = gdal.Open(imgpath)
        gt = tif.GetGeoTransform()
        #print(gt)
        labelname = line.split(',')[-1].split('|')[0]
        #print(labelname)
        x_min = gt[0]-39000000  #qu chu daihao yingxiang
        #x_min = gt[0]
        x_size = gt[1]
        y_min = gt[3]
        y_size = gt[5]
        p0_x = float(pixel_loc[0])*x_size + x_min
        p0_y = float(pixel_loc[1])*y_size + y_min
        p1_x = float(pixel_loc[2])*x_size + x_min
        p1_y = float(pixel_loc[1])*y_size + y_min
        p2_x = float(pixel_loc[2])*x_size + x_min
        p2_y = float(pixel_loc[3])*y_size + y_min
        p3_x = float(pixel_loc[0])*x_size + x_min
        p3_y = float(pixel_loc[3])*y_size + y_min
        writer.writerow([p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, labelname])