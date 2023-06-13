import cv2 as cv
import numpy as np

img = cv.imread('/home/yf/disk/camera/test/Camera_2020_05_20_09_53_11.jpg')
path = '/home/yf/disk/camera/my.jpg'
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

cv.imwrite(path,ret)