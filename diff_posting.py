from tifffile import imread, imsave
import numpy as np
import cupy as cp
import scipy.ndimage as ndi
from skimage import measure
from tqdm import tqdm


prefix = '/home/yf/Documents/mmsegmentation/work_dirs/whu_ocrnet_0.6/pred'
img1_path= os.path.join(prefix,'afterpred.tif')
#'/home/yf/disk/whu/part4/bcdd/two_period_data/combine_2000/afterpred.tif'
#'/home/yf/disk/qgis34-files/object/after_pred.tif'
img2_path=os.path.join(prefix,'beforepred.tif')
diff_path=os.path.join(prefix,'refine_diff2.tif')
img1=np.array(imread(img1_path))
img2=np.array(imread(img2_path))


ori_w, ori_h = img2.shape
label1=measure.label(img1,connectivity=2)
label2=measure.label(img2,connectivity=2)
# label1=cp.asarray(label1)
# label2=cp.asarray(label2)
tmp=cp.zeros_like(label1)
diff=img1-img2
diff=cp.asarray(diff)
collect=[1,-1]
stride=10000
pad_w = int(ceil(ori_w / stride) * stride) - ori_w
pad_h = int(ceil(ori_h / stride) * stride) - ori_h
# print(pad_w, pad_h)
# print(ceil(ori_w/stride), ceil(ori_h/stride))lllllllllllll
# print(ori_w/stride, ori_h/stride)
new_w = ori_w + pad_w
new_h = ori_h + pad_h
pad_img = cp.zeros(shape=(new_w, new_h),dtype=int8)
num_w = int(new_w / stride)
num_h = int(new_h / stride)

for w_id in range(num_w):
    for h_id in range(num_h):
        tmp=diff[w_id*stride:w_id*stride+stride, h_id*stride:h_id*stride+stride].copy()
        diff_tmp = diff[w_id*stride:w_id*stride+stride, h_id*stride:h_id*stride+stride]
        for val in collect:
            # tmp=diff.copy()
            tmp[tmp!=val]=0
            tmp[tmp==val]=1
            # tmp_label=measure.label(tmp,connectivity=2)
            # regin_ids = np.unique(tmp_label)
            if val == 1:
                label = label1[w_id*stride:w_id*stride+stride, h_id*stride:h_id*stride+stride]
            else:
                label = label2[w_id*stride:w_id*stride+stride, h_id*stride:h_id*stride+stride]
            # for regin_id in tqdm(regin_ids):
            #     if regin_id!=0:
            #         index = np.where(tmp_label==regin_id)
            #         #diff_img = tmp_label[index]
            #         #print(index)
            #         x = index[0][0]
            #         y= index[1][0]
            #         diff_area = len(index[0])
            #         label_id = label[x,y]
            #         tmp_index = np.where(label==label_id)
            #         img_area = len(tmp_index[0])
            #         if diff_area/img_area<0.1:
            #             diff[index]=0
            label = cp.asarray(label)
            regin_ids = cp.unique(label)
            for regin_id in tqdm(regin_ids):
                if regin_id!=0:
                    index = cp.where(label == regin_id)
                    area = len(index[0])
                    tmp_diff = tmp[index]
                    ones_sum = len(cp.where(tmp_diff==1)[0])
                    if ones_sum/area<0.2:
                        diff_tmp[index]  = 0


diff[diff==-1]=125
diff[diff==1]=255
imsave(diff_path, diff)