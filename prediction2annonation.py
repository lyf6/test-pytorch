import os
from time import CLOCK_PROCESS_CPUTIME_ID
import PIL.Image as Image
import numpy as np
import cv2
import base64
import json
from skimage import measure
from tqdm import tqdm

pred_dir = '/home/yf/disk/res/objs'
json_dir = '/home/yf/disk/res/json_dir'
img_dir = '/home/yf/disk/res/JPEGImages'
area_threshold = 90000 #90000 for farmland
length_threshold = 0.005 #0.005 for farmland
erode_kernel = 51 #51 for farmland
img_ls = os.listdir(img_dir)
div_nums = 3
clssids = {15:'farmland', 30:'none_farmland'}
stride = 100

for img_name in img_ls:
    pred_img_path = os.path.join(pred_dir, img_name.split('.')[0]+'pred.tif')
    json_file = os.path.join(json_dir, img_name.split('.')[0]+'.json')
    img_path = os.path.join(img_dir, img_name)
    dumpy = {}
    dumpy["version"] = "4.6.0"
    dumpy["flags"] = {}
    img = np.array(Image.open(pred_img_path))
    height, width = img.shape
    h_stride = int(height/div_nums)
    w_stride = int(width/div_nums)
    for i in range(div_nums):
        img[(i+1)*h_stride:(i+1)*h_stride+stride,:] = 255 
    for i in range(div_nums):
        img[:, (i+1)*w_stride:(i+1)*w_stride+stride] = 255
    shapes = []
    for clssid in clssids.keys():
        tmp_img = img.copy()
        one_index = tmp_img == clssid
        zero_index = tmp_img != clssid
        tmp_img[one_index] = 1
        tmp_img[zero_index] = 0
        tmp_img = tmp_img.astype(np.uint8)
        if erode_kernel is not None:       
            kernel = np.ones((erode_kernel,erode_kernel),np.uint8)
            tmp_img = cv2.erode(tmp_img,kernel)
        contours, _ = cv2.findContours(tmp_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # tmp_label = measure.label(tmp_img, connectivity=2)
        # proper_tmp = measure.regionprops(tmp_label)
        # num_region = len(proper_tmp)
        # for re_id in tqdm(range(num_region)):
        #     sub_region_aera =  proper_tmp[re_id].area
        #     if sub_region_aera > 40000:
        #         centroid = proper_tmp[re_id].centroid
        #         tmp = {}
        #         tmp["label"] = clssids[clssid]
        #         tmp["line_color"] = None
        #         tmp["fill_color"] = None
        #         tmp['points'] = [[centroid[0]-scale, centroid[1]-scale], [centroid[0]-scale, centroid[1]+scale], \
        #             [centroid[0]+scale, centroid[1]+scale], [centroid[0]+scale, centroid[1]-scale]]
        #         tmp['shape_type'] = 'polygon'
        #         tmp['flags'] = {}
        #         shapes.append(tmp)
        for cnt in contours:
            area =cv2.contourArea(cnt)
            if area > area_threshold:
                cnt = cv2.approxPolyDP(cnt,length_threshold*cv2.arcLength(cnt,True),True)
                cnt = np.squeeze(cnt).astype(np.float)
                tmp = {}
                tmp["label"] = clssids[clssid]
                tmp["line_color"] = None
                tmp["fill_color"] = None
                tmp['points'] = [[point[0], point[1]] for point in cnt]
                tmp['shape_type'] = 'polygon'
                tmp['flags'] = {}
                shapes.append(tmp)
    #cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if len(shapes) > 0:
        dumpy['shapes'] = shapes
        # dumpy['lineColor'] = [0, 255, 0, 128]
        # dumpy['fillColor'] = [255, 0, 0, 128]
        dumpy['imagePath'] = img_name
        with open(img_path, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
        dumpy['imageData'] = imageData
        dumpy['imageHeight'] = height
        dumpy['imageWidth'] = width
        with open(json_file, 'w') as outfile:
            outfile.write(json.dumps(dumpy, indent=2))
                    