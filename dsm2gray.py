from osgeo import gdal


dsm= "/home/yf/disk/jinzhongqiao/Production_3_DSM_merge.tif"
gray= "/home/yf/disk/jinzhongqiao/DSM_gray.tif"
img_handle = gdal.Open(dsm)
img = img_handle.ReadAsArray()

min_value = -6
max_value = 1.5
# print(min_value)
index = img<min_value
img[index] = min_value
index = img>max_value
img[index] = max_value
img = img - min_value
stand = 255/(max_value-min_value)
img = img*stand
cv2.imwrite(gray, img.astype(np.uint8))