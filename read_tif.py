import gdal
from math import *
import csv

path = '/home/yf/disk/top.tif'
tif = gdal.Open(path)
gt = tif.GetGeoTransform()
left_top_longitude = gt[0]
left_top_latitude = gt[3]
pixel_longitude = gt[1]
pixel_latitude = gt[5]
ori_h = tif.RasterXSize
ori_w= tif.RasterYSize
weight=2000
height=2000
stride=2000
pad_w = (ceil(ori_w / stride)) * stride - ori_w
pad_h = (ceil(ori_h / stride)) * stride - ori_h
new_w = ori_w + pad_w
new_h = ori_h + pad_h
num_w = int(new_w / weight)
num_h = int(new_h / height)
csv_path = '/home/yf/disk/result.csv'
file = open(csv_path)
csv_read = csv.reader(file)
write_path = '/home/yf/disk/lon_lat.csv'
wt = open(write_path,'a+')
csv_write = csv.writer(wt)


for line in csv_read:
    tmp = line
    img_id = int(tmp[0])
    row = img_id // num_h
    cow = img_id % num_h
    start_y = row * stride
    start_x = cow * stride
    left_top_y = int(tmp[2]) + start_y
    left_top_x = int(tmp[1]) + start_x
    right_bottom_y = int(tmp[4]) + start_y
    right_bottom_x = int(tmp[3]) + start_x
    left_top_lat = left_top_latitude + left_top_y * pixel_latitude
    left_top_lon = left_top_longitude + left_top_x * pixel_longitude
    right_bottom_lat = left_top_latitude + right_bottom_y * pixel_latitude
    right_bottom_lon = left_top_longitude + right_bottom_x * pixel_longitude
    wt_line = [str(left_top_lon), str(left_top_lat), str(right_bottom_lon), str(right_bottom_lat),tmp[5]]
    csv_write.writerow(wt_line)