import solaris as sol
from solaris.data import data_dir
import os
import skimage
import geopandas as gpd
from matplotlib import pyplot as plt
from shapely.ops import cascaded_union
import numpy as np
from solaris.utils import io
from PIL import Image
"generate mask for road and 8bit image"


root='/home/yf/disk/spacenet3'
objects='spacenetroads'
type='Sharpen_'
city_list=['Vegas', 'Paris', 'Shanghai', 'Khartoum']
maskpath=root+'/2mmasks'
eight_bit_path=root+'/8bit_ps'

folder = os.path.isdir(maskpath)
if not folder:
    os.makedirs(maskpath)

folder = os.path.isdir(eight_bit_path)
if not folder:
    os.makedirs(eight_bit_path)

for id, city in enumerate(city_list):
    specific_path='AOI_'+str(id+2)+'_'+city+'_Roads_Train'
    city_path=os.path.join(root,specific_path)
    imgpath=os.path.join(city_path, 'RGB-PanSharpen')
    img_ls=os.listdir(imgpath)
    #geo_ls=os.listdir(os.path.join())
    for img in img_ls:
        img_path=os.path.join(imgpath,img)
        # print(img_path)
        eight_bit=io.imread(img_path, make_8bit=True,rescale=True)
        tmp=img.split(type)[1].split('.tif')[0]
        #print(eight_bit.shape)
        tar_img_path = os.path.join(eight_bit_path, tmp +'.jpg')

        geo_path='geojson/'+'spacenetroads/'+objects+'_'+tmp+'.geojson'
        geo_path=os.path.join(city_path,geo_path)
        try:
            gpd_file = gpd.read_file(geo_path)
            road_mask = sol.vector.mask.road_mask(gpd_file,
                                                reference_im=img_path,
                                                width=2, meters=True, burn_value=1)
            tar_mask_path=os.path.join(maskpath,tmp +'.png')
            skimage.io.imsave(tar_mask_path,road_mask)
            skimage.io.imsave(tar_img_path, eight_bit)
        except:
            # road_mask = sol.vector.mask.road_mask(gpd_file,
            #                                     reference_im=img_path,
            #                                     width=3, meters=True, burn_value=1)
            continue
        #     image = Image.open(img_path)
        #     height, width = image.height, image.width
        #     road_mask = np.zeros(shape=(height, width), dtype=np.uint8)
        #     print('ddddd ' + img_path)
        # if road_mask is not None:
        #     tar_mask_path=os.path.join(maskpath,tmp +'.png')
        #     skimage.io.imsave(tar_mask_path,road_mask)
        #     skimage.io.imsave(tar_img_path, eight_bit)