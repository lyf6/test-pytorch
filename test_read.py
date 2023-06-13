from tifffile import imread
import numpy as np
import matplotlib

path = '/home/yf/disk/whu/part4/bcdd/two_period_data/5000/pred/after_3_0.tif'
#save_path = '/home/yf/disk/whu/part4/bcdd/two_period_data/results/after_0_18.png'
tmp_img = np.array(imread(path))
print(tmp_img)
# tmp_img=Image.fromarray(tmp_img)
#matplotlib.image.imsave(save_path, tmp_img)