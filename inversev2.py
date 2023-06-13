import gdal
from math import *
import csv
from PIL import Image, ImageDraw
import numpy as np
Image.MAX_IMAGE_PIXELS = None


path = '/home/yf/disk/top.tif'
img = np.array(Image.open(path))
ori_w, ori_h, _= img.shape
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
img = Image.open(path)
draw = ImageDraw.Draw(img)
save_path = '/home/yf/disk/res.jpg'
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
    draw.rectangle([left_top_x, left_top_y, right_bottom_x, right_bottom_y], outline=(0, 255, 0), width=10)

img.save(save_path)