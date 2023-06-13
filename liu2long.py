import os
import gdal
import csv
import re
# -*- coding:GBK -*-

file = 'construction_site.txt'
#target_file = 'greenhouse.csv'
source_imgs = '/home/yf/disk/testdata/2019xq'
fp = open(file, encoding='GBK')
cla_name = ['construction_site', 'bareland', 'greenhouse']
target_csv = 'construction_site.csv'
myFile = open(target_csv,'w',encoding='utf8',newline='')
writer = csv.writer(myFile)
lines = fp.read().splitlines()
for line in lines:
    tmp =line.split(' ')
    #print(tmp)
    img_name = tmp[0].split('JpegImages')[1].split('.jpg')[0]+'.tif'
    #print(tmp[1:5][0])
    #pixel_loc = re.findall(r"\d+\.?\d*", tmp[1:5])

    #print(pixel_loc)
    labelname = cla_name[int(tmp[6].split('=')[1])]
    pixel_loc = re.findall(r"\d+\.?\d*",line)[3:7]
    print(pixel_loc)
    #print(img_name)
    img = source_imgs+img_name
    #print(img)
    tif = gdal.Open(img)

    print(tif)
    gt = tif.GetGeoTransform()
    print(gt)
    x_min = gt[0]-39000000
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

