import cv2
import os
import numpy as np
import csv
from PIL import Image
import shapefile
from tqdm import tqdm
import gdal

def draw_approx_hull_polygon(img, cnts):
    # img = np.copy(img)
    w, h = img.shape
    img = np.zeros(shape=(w,h, 3), dtype=np.uint8)

    cv2.drawContours(img, cnts, -1, (255, 0, 0), 2)  # blue

    # epsilion = img.shape[0]/32
    # approxes = [cv2.approxPolyDP(cnt, epsilion, True) for cnt in cnts]
    # cv2.polylines(img, approxes, True, (0, 255, 0), 2)  # green
    #
    # hulls = [cv2.convexHull(cnt) for cnt in cnts]
    # cv2.polylines(img, hulls, True, (0, 0, 255), 2)  # red

    # 我个人比较喜欢用上面的列表解析，我不喜欢用for循环，看不惯的，就注释上面的代码，启用下面的
    # for cnt in cnts:
    #     cv2.drawContours(img, [cnt, ], -1, (255, 0, 0), 2)  # blue
    #
    #     epsilon = 0.01 * cv2.arcLength(cnt, True)
    #     approx = cv2.approxPolyDP(cnt, epsilon, True)
    #     cv2.polylines(img, [approx, ], True, (0, 255, 0), 2)  # green
    #
    #     hull = cv2.convexHull(cnt)
    #     cv2.polylines(img, [hull, ], True, (0, 0, 255), 2)  # red
    return img


#dir = '/home/yf/disk/output/myann/seg_hrnet_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484x2/test_results'
# imgdir = '/home/yf/disk/myannotated_cd/test'
##imglist = os.listdir(dir)


cls_id = {0: 'building'}
w = shapefile.Writer(shapefile.POLYGON)
w.autoBalance = 1
w.field('BUFF_DIST', 'C', '40')
# for img_name in imglist:
    
imgpath = '/home/yf/disk/testdata/bd/bdfrgbp_pred.tif' #os.path.join(dir, img_name)
tareget_shape = '/home/yf/disk/testdata/bd/bdfrgbp_pred'
#print(img_name)
tif = gdal.Open(imgpath)
gt = tif.GetGeoTransform()
img = np.array(tif.ReadAsArray())
print('processing '+ imgpath)
x_min = gt[0]
x_size = gt[1]
y_min = gt[3]
y_size = gt[5]

for key, value in cls_id.items():
    index = img==key
    tmp_img = np.zeros_like(img)
    tmp_img[index] = 255
    ret, thresh = cv2.threshold(tmp_img, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refine_cnt = []
    for cnt in contours:
        area =cv2.contourArea(cnt)
        #print(cnt)
        if area >30:
            refine_cnt.append(cnt)
    #tmp_str = ''
    if len(refine_cnt)>0:
        for cnt in tqdm(refine_cnt):
            #print(cnt)
            #tmp_str = img_name + ' ' + value
            #print(tmp_str)
            point_list = []
            polygon_list = []
            point = []
            for loc in cnt:
                x = loc[0][0]*x_size + x_min
                y = loc[0][1]*y_size + y_min
                point.append(float(x))
                point.append(float(y))
                point_list.append(point)
                point = []
            polygon_list.append(point_list)
            w.poly(parts=polygon_list)
            w.record(value)
w.save(tareget_shape)            
                #tmp_str = tmp_str + ' ' + '(' + str(x) + ', ' + str(y) + ')'
            #print(tmp_str)
            #writer.writerow([tmp_str])



# path = '/home/yf/disk/output/sjd/seg_hrnet_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484x2/refine_test_results/tj_120111xq201902dom_348.png'
# img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
# #print(img)
# ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# refine_cnt = []
# for cnt in contours:
#     area =cv2.contourArea(cnt)
#     if area >100:
#         refine_cnt.append(cnt)





# img = draw_approx_hull_polygon(img, refine_cnt)
# cv2.imwrite('test2.png', img)
