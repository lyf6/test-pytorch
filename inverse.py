import gdal
from math import *
import csv
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None

path = '/home/yf/disk/top.tif'
tif = gdal.Open(path)
gt = tif.GetGeoTransform()
left_top_longitude = gt[0]
left_top_latitude = gt[3]
pixel_longitude = gt[1]
pixel_latitude = gt[5]
img = Image.open(path)
draw = ImageDraw.Draw(img)

csv_path = '/home/yf/disk/lon_lat.csv'
file = open(csv_path)
csv_read = csv.reader(file)

save_path = '/home/yf/disk/res.jpg'
#img.show()

for line in csv_read:
    tmp = line

    left_top_lat = float(tmp[1])
    left_top_lon = float(tmp[0])
    right_bottom_lat = float(tmp[3])
    right_bottom_lon = float(tmp[2])
    left_top_y = (left_top_lat - left_top_latitude)/pixel_latitude
    left_top_x = (left_top_lon - left_top_longitude)/pixel_longitude
    right_bottom_y = (right_bottom_lat - left_top_latitude )/pixel_latitude
    right_bottom_x = (right_bottom_lon - left_top_longitude)/pixel_longitude
    draw.rectangle([left_top_x, left_top_y, right_bottom_x, right_bottom_y], outline=(0,255,0), width=10)

img.save(save_path)



