from PIL import Image
path='/home/yf/Documents/data/RS/images/train/GF2_PMS2__20160510_L1A0001573999-MSS2 (2).tif'
tif = Image.open(path) # open tiff file in read mode
# read an image in the currect TIFF directory as a numpy array

print('finished')