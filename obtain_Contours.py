import cv2
import os
import numpy as np
import csv

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


dir = '/home/yf/disk/output/sjd_v2/seg_hrnet_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484x2/refine_test_results'
imglist = os.listdir(dir)

target_csv = 'pixel_sjd.csv'
myFile = open(target_csv,'w',newline='')
writer = csv.writer(myFile)

for img_name in imglist:
    imgpath = os.path.join(dir, img_name)
    #print(img_name)
    img = cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refine_cnt = []
    for cnt in contours:
        area =cv2.contourArea(cnt)
        #print(cnt)
        if area >100:
            refine_cnt.append(cnt)
    tmp_str = ''
    if len(refine_cnt)>0:
        for cnt in refine_cnt:
            #print(cnt)
            tmp_str = img_name
            #print(tmp_str)
            for loc in cnt:
                x = loc[0][0]
                y = loc[0][1]
                tmp_str = tmp_str + ' ' + '(' + str(x) + ', ' + str(y) + ')'
            #print(tmp_str)
            writer.writerow([tmp_str])



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