import os
from PIL import Image
import json
import re
import random
import numpy as np
import cv2
import base64
import multiprocessing
from functools import partial

def obtain_cnts(img, minarea=300):
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refine_cnt = []
    for cnt in contours:
        area =cv2.contourArea(cnt)
        #print(cnt)
        if area > minarea:
            refine_cnt.append(cnt)
    return refine_cnt


def write_json(img_name, result_dir, ori_imgdir, label_name, json_dir):
    img_path = os.path.join(result_dir, img_name)
    cls_img = Image.open(img_path)
    h, w = cls_img.size
    #cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    json_file = os.path.join(json_dir,img_name.split('.')[0]+'.json')
    dumpy = {}
    dumpy["version"] = "3.16.7"
    dumpy["flags"] = {}
    shapes =[]
    for cls_id in range(len(label_name)):
        np_img = np.array(cls_img)
        index_0 = np_img == cls_id # wanted class
        index_1 = np_img != cls_id
        np_img[index_0] = 255
        np_img[index_1] = 0
        cnts = obtain_cnts(np_img)
        if len(cnts)>0:
            imgname = os.path.join(ori_imgdir, img_name.split('.png')[0]+'.tif')
            labelname = label_name[cls_id]
            for cnt in cnts:
                epsilon = 0.01*cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, epsilon, True)
                cnt = np.squeeze(cnt, axis=1).tolist()
                #print(cnt)
                tmp = {}
                tmp["label"] = labelname
                tmp["line_color"] = None
                tmp["fill_color"] = None
                tmp['points'] = cnt
                tmp['shape_type'] = 'polygon'
                tmp['flags'] ={}
                shapes.append(tmp)
    if len(shapes)>0:
        dumpy['shapes'] = shapes
        dumpy['lineColor'] = [0,255,0,128]
        dumpy['fillColor'] = [255,0,0,128]
        dumpy['imagePath'] = imgname
        with open(imgname, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
        dumpy['imageData'] = imageData
        dumpy['imageHeight'] = h
        dumpy['imageWidth'] = w
        with open(json_file, 'w') as outfile:
            outfile.write(json.dumps(dumpy, indent=2))

if __name__ == "__main__":

    result_dir = '/home/yf/disk/output/sjd/seg_hrnet_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484x2/refine_test_results'
    ori_imgdir = '/home/yf/disk/testdata/2019jh'
    label_name = ['other', 'sjd']
    imglist = os.listdir(result_dir)
    json_dir = '/home/yf/disk/testdata/2019jh_json'
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    part_func = partial(write_json, result_dir = result_dir , ori_imgdir = ori_imgdir, label_name = label_name, json_dir = json_dir)

    #p = multiprocessing.Process(target = write_json, args = (result_dir,ori_imgdir,label_name,json_dir))
    pool = multiprocessing.Pool(processes = 12)
    #p.start()
    tmp =[]
    for img in imglist:
        if 'jh2019' in img:
            tmp.append(img)

    pool.map(part_func, tmp)
