import os
from PIL import Image
import json
import re
import random
import numpy as np
import base64

source_json = '/home/yf/disk/camera/pred8/IP PTZ Camera_2020_05_20_09_53_42.json'

target_json_dir  = '/home/yf/disk/camera/pred8'
new_target_json_dir  = '/home/yf/disk/camera/new_pred8'
json_list = os.listdir(target_json_dir)
data=open(source_json)
data = json.load(data)
wanted_points = []
search_part = data["shapes"]
for shape in search_part:
    if shape['label'] == 'other':
        wanted_points.append(shape)



for jsons in json_list:
    json_path = os.path.join(target_json_dir, jsons)
    print(json_path)
    tarfet_data = open(json_path)
    tarfet_data = json.load(tarfet_data)

    stroe_dic = []
    for id, shape in enumerate(tarfet_data['shapes']):
        if shape['label'] != 'other':
            stroe_dic.append(shape)
    del tarfet_data['shapes']

    stroe_dic = stroe_dic + wanted_points
    tarfet_data['shapes'] = stroe_dic

    tmp_json = os.path.join(new_target_json_dir, jsons)
    with open(tmp_json, 'w') as outfile:
        json.dump(tarfet_data, outfile)
