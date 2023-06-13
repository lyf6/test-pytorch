import os
import sys
import gdal
import numpy as np
import ogr
import osr
import scipy.misc
from skimage.draw import polygon
from skimage import io
from skimage import measure
from glob import glob
from PIL import Image
from skimage import measure
from math import *
#import tifffile
#from combine import array2raster
from os.path import basename
import datetime
from pycococreatortools import pycococreatortools
import json

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--imgdir', help='path to img dir', default='/home/yf/disk/buildings/whu/part4/bcdd/two_period_data/div_images')
    parser.add_argument('--anndir', help='the dir to ann dir', default='/home/yf/disk/buildings/whu/part4/bcdd/two_period_data/div_labels')
    parser.add_argument('--id_class_map_file', help='the file of id_class_map_file', default='./class_list.txt')
    parser.add_argument('--workdir', help='the dir to save json', default='/home/yf/disk/buildings/whu/part4/bcdd/two_period_data')
    args = parser.parse_args()
    return args


def main():
    #listParams = ['imgdir', 'anndir', 'id_class_map_file', 'workdir']
    # print(len(sys.argv))
    # if len(sys.argv) < len(listParams):
    #     sys.exit('Usage: ' + sys.argv[0] + ' ' + '\n'.join(listParams))

    # index = 1
    args  = parse_args()
    imgdir = args.imgdir
   
    anndir = args.anndir
    
    id_class_map_file = args.id_class_map_file
    
    workdir = args.workdir

    INFO = {
        "description": "object detection dataset",
        "url": "",
        "version": "0.1.0",
        "year": 2020,
        "contributor": "Yanfei Liu",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    LICENSES = [
        {
            "id": 1,
            "name": "",
            "url": ""
        }
    ]
    CATEGORIES = [ ]
    #print(listParams)
    op_file = open(id_class_map_file)
   
    #print(op_file)
    lines = op_file.readlines()
    tmp = {}
    record_id_cls = {}
    print(lines)
    for line in lines:
        line = line.strip()
        id_class = line.split(':')
        tmp['id'] = int(id_class[0])
        tmp['name'] = id_class[1]
        tmp['supercategory'] = 'object'
        CATEGORIES.append(tmp)
        record_id_cls[int(id_class[0])] = id_class[1]
        tmp = {}

    print(record_id_cls)
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    img_list = os.listdir(imgdir)
    #print(img_list)
    for img in img_list:
        #print(img)
        img_name = os.path.join(imgdir, img)
        image = Image.open(img_name)
        image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(img_name), image.size)
        coco_output["images"].append(image_info)

        if os.path.exists(os.path.join(anndir, img)):
            mask = np.asarray(Image.open(os.path.join(anndir, img)))
            label_list = np.unique(mask)
            for label_id in label_list:
                if label_id in record_id_cls.keys():
                    class_id = label_id
                    category_info = {'id': int(class_id), 'is_crowd': False}
                    binary_mask = np.asarray(mask==class_id).astype(np.uint8)
                    tmp_label = measure.label(binary_mask, connectivity=2)
                    proper_tmp = measure.regionprops(tmp_label)
                    num_region = len(proper_tmp)
                    for regin_id in range(num_region):
                        result = np.zeros_like(mask)
                        real_id = regin_id + 1
                        index = tmp_label==real_id
                        result[index] = 1
                        #print(result)
                        annotation_info = pycococreatortools.create_annotation_info(
                            segmentation_id, image_id, category_info, result,
                            image.size, tolerance=2)
                        if annotation_info is not None:
                            coco_output["annotations"].append(annotation_info)
                        segmentation_id = segmentation_id + 1
        image_id = image_id + 1

    annfile = os.path.join(workdir, 'instance.json')
    with open(annfile, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)

if __name__ == '__main__':

    main()