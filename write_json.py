import os
from PIL import Image
import json
import re
import random
import numpy as np

# category_list =  {"dust_cover_net":0, "tower_crane":1,"earthmoving_truck":2,"middle_bus":3,
#                 "buildings_construction":4,"construction_site":5,"crane":6,
#                 "excavating_machinery":7,"building_materials":8,"other_machinery":9,
#                 "private_car":10,"scaffolding":11,"person":12,"three_wheel":13}
category_list = {'construction_site':0}


category_count=0

def write2json(outfile, annation_list):
    INFO = {
        "description": "the data from tigis",
        "url": " ",
        "version": "1.0",
        "year": 2019,
        "contributor": "bd",
        "date_created": "2019/7/1"
    }
    global category_count
    categories = []
    images=[]
    #outfile='D:\\data\\lllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllltk\\annotations.json'
    annotations = []
    instance_count=0
    for annation_file in annation_list:
        filepath = os.path.join(annations_dir, annation_file)
        if os.path.isfile(filepath):
            data=open(filepath)
            data = json.load(data)
            tmp_img={}
            tmp_img['license']=1
            tmp_img['file_name'] = data['imagePath'].split('\\')[-1]
            tmp_img['height']=data['imageHeight']
            tmp_img['width']=data['imageWidth']
            tmp_img['id']=data['imagePath'].split('\\')[-1]
            images.append(tmp_img)
            all_instance=data['shapes']
            for instance in all_instance:
                annotation={}
                annotation['area']=500
                annotation['iscrowd']=0
                annotation['image_id']=data['imagePath'].split('\\')[-1]
                if instance['label'] in category_list:
                    category_id=category_list[instance['label']]
                else:
                    category_list[instance['label']]=category_count
                    category_id=category_list[instance['label']]
                    category_count+=1
                    print('wrong here')
                annotation['category_id'] = category_id
                instance_points=instance['points']
                shape_type=instance['shape_type']
                segmentation = []
                tmp_segmentation = []
                polygon=[]
                x_list=[]
                y_list=[]
                for point in instance_points:
                    x_list.append(point[0])
                    y_list.append(point[1])
                    tmp_segmentation.append(point[0])
                    tmp_segmentation.append(point[1])
                if shape_type=='rectangle':
                    polygon.append(tmp_segmentation[0])
                    polygon.append(tmp_segmentation[1])
                    polygon.append(tmp_segmentation[2])
                    polygon.append(tmp_segmentation[1])
                    polygon.append(tmp_segmentation[2])
                    polygon.append(tmp_segmentation[3])
                    polygon.append(tmp_segmentation[0])
                    polygon.append(tmp_segmentation[3])
                    segmentation.append(polygon)
                else:
                    #tmp_segmentation=np.array(tmp_segmentation)
                    segmentation.append(tmp_segmentation)
                annotation['segmentation']=segmentation
                annotation['id']=instance_count
                instance_count+=1
                bbox=[]
                bbox.append(min(x_list))
                bbox.append(min(y_list))
                bbox.append(max(x_list)-min(x_list))
                bbox.append(max(y_list) - min(y_list))
                annotation['bbox']=bbox
                annotations.append(annotation)

    for label in category_list:
        tmp={}
        tmp['supercategory']='gd'
        tmp['id']=category_list[label]
        tmp['name']=label
        categories.append(tmp)

    tk={}
    tk['info']=INFO
    tk['images']=images
    tk['annotations']=annotations
    tk['categories']=categories

    with open(outfile, 'w') as outfile:
        json.dump(tk, outfile)

annations_dir = '/home/yf/disk/tj/json'
tmp_annation_list = os.listdir(annations_dir)
trainfile='/home/yf/disk/tj/annotations/train.json'
testfile='/home/yf/disk/tj/annotations/val.json'
imgdir='/home/yf/disk/tj/imgs/'

annation_list=[]
for annation in tmp_annation_list:
    id = annation.split('.')[0]
    if os.path.exists(imgdir+id+'.jpg') or os.path.exists(imgdir+id+'.png') :
        annation_list.append(annation)




percent=1.0
train_num=int(len(annation_list)*percent)
random.shuffle(annation_list)
train_list=annation_list[:train_num]
test_list=annation_list[train_num:]
write2json(trainfile,train_list)
write2json(testfile,test_list)