from mmseg.apis import inference_segmentor, init_segmentor, my_inference_segmentor
import mmcv
import os
from PIL import Image
import numpy as np
from tifffile import imsave

import torch
from torchvision.models import resnet18




dir='/home/yf/Documents/mmsegmentation/'

config_file =dir+'configs/ocrnet/ocrnet_hr48_512x512_10k_myann.py'
    #'configs/ocrnet/ocrnet_hr48_512x512_50k_whu.py'
    #'configs/ocrnet/ocrnet_hr48_512x512_10k_myann.py'
checkpoint_file = dir+'work_dirs/myann/latest.pth'
# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# # test a single image and show the results
# img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
# result = inference_segmentor(model, img)
# # visualize the results in a new window
# model.show_result(img, result, show=True)
# # or save the visualization results to image files
# model.show_result(img, result, out_file='result.jpg')

test_dir = '/home/yf/Documents/mmsegmentation/data/whu/img_dir/test'
save_dir = '/home/yf/disk/whu/part4/bcdd/two_period_data/results'
img_list = os.listdir(test_dir)
#for segmap image
# for img_id in img_list:
#
#     img = os.path.join(test_dir, img_id)
#     result = inference_segmentor(model, img)[0]
#     result = result.astype(np.uint8)
#     save_img = Image.fromarray(result)
#     save_path = os.path.join(save_dir, img_id)
#     print(save_path)
#     save_img.save(save_path)

#for probability map
for img_id in img_list:
    img = os.path.join(test_dir, img_id)
    result = my_inference_segmentor(model, img, logit=True)[0]
    result = np.squeeze(result)
    print(result.shape)
    save_path = os.path.join(save_dir, img_id)
    print(save_path)
    imsave(save_path, result)

# test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_segmentor(model, frame)
#     model.show_result(frame, result, wait_time=1)