from tifffile import imread, imsave
import numpy as np
import scipy.ndimage as ndi
from skimage import measure
from tqdm import tqdm
import gdal, ogr, os, osr

def get_confusion_matrix(label, pred_label, num_class):
    """
    Calcute the confusion matrix by given label and pred
    """

    label = label.reshape(-1)
    pred_label = pred_label.reshape(-1)
    index = (label * num_class + pred_label).astype('int32')

    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

prefix = '/home/yf/Documents/mmsegmentation/work_dirs/whu_ocrnet_0.6/pred'
pred_path=os.path.join(prefix,'refine_diffv4.tif')
#pred_path='/home/yf/disk/whu/part4/bcdd/two_period_data/combine/diff.tif'
#'/home/yf/disk/myannotated_cd/prod_comb/refine_diff.tif'
#'/home/yf/disk/whu/part4/bcdd/two_period_data/combine/refine_diff.tif'
#'/home/yf/disk/myannotated_cd/prod_comb/refine_diff.tif'
#'/home/yf/disk/qgis34-files/object/refine_diff.tif'
gt_path='/home/yf/disk/whu/part4/bcdd/two_period_data/change_label/change_label.tif'
#'/home/yf/disk/myannotated_cd/2017-2019val/2019-2017_change_remap.tif'
 #'/home/yf/disk/whu/part4/bcdd/two_period_data/change_label/change_label.tif'
#'/home/yf/disk/myannotated_cd/2017-2019val/2019-2017_change_remap.tif'
#'/home/yf/disk/qgis34-files/object/change_label.tif'

tif = gdal.Open(pred_path)
pred = np.array(tif.ReadAsArray())
index = pred>0
pred[index]=1

tif = gdal.Open(gt_path)
gt = np.array(tif.ReadAsArray())


num_class = 2
confusion_matrix  = get_confusion_matrix(gt, pred, num_class)
pos = confusion_matrix.sum(1)
res = confusion_matrix.sum(0)
diag = confusion_matrix.diagonal()
recal = diag / pos
precision = diag / res
f1 = 2 * recal * precision / (recal + precision)
print('recall is: ')
print(recal)
print('precision is: ')
print(precision)
print('f1 is:')
print(f1)