import solaris as sol
from solaris.data import data_dir
import os
import skimage
import geopandas as gpd
from matplotlib import pyplot as plt
from shapely.ops import cascaded_union
import numpy as np
from solaris.utils import io

"generate mask for building and 8bit image"


root='/home/yf/disk/spacenet2'
objects='buildings'
type='Sharpen_'
city_list=['Vegas', 'Paris', 'Shanghai', 'Khartoum']
maskpath=root+'/masks'
eight_bit_path=root+'/8bit_ps'

folder = os.path.isdir(maskpath)
if not folder:
    os.makedirs(maskpath)

folder = os.path.isdir(eight_bit_path)
if not folder:
    os.makedirs(eight_bit_path)

for id, city in enumerate(city_list):
    specific_path='AOI_'+str(id+2)+'_'+city+'_Train'
    city_path=os.path.join(root,specific_path)
    imgpath=os.path.join(city_path, 'RGB-PanSharpen')
    img_ls=os.listdir(imgpath)
    #geo_ls=os.listdir(os.path.join())
    for img in img_ls:
        img_path=os.path.join(imgpath,img)
        print(img_path)
        eight_bit=io.imread(img_path, make_8bit=True,rescale=True)
        tmp=img.split(type)[1].split('.tif')[0]
        #print(eight_bit.shape)
        tar_img_path = os.path.join(eight_bit_path, img)
        skimage.io.imsave(tar_img_path, eight_bit)
        geo_path='geojson/'+'buildings/'+objects+'_'+tmp+'.geojson'
        geo_path=os.path.join(city_path,geo_path)
        fbc_mask = sol.vector.mask.df_to_px_mask(df=geo_path,
                                                 channels=['footprint', 'boundary', 'contact'],
                                                 reference_im=img_path,
                                                 boundary_width=5, contact_spacing=10, meters=True)



        tar_mask_path=os.path.join(maskpath,img)

        skimage.io.imsave(tar_mask_path,fbc_mask)