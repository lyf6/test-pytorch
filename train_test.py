import os


source='/home/yf/disk/object_detection/uavcar/allinone/JPEGImages/'
txt='/home/yf/disk/object_detection/uavcar/allinone/trainval.txt'
filelist=os.listdir(source)
target_file=open(txt,'w')
# for file in filelist:
#     #tmp='div_train/images/'+file+'   '+'div_train/masks/'+file.split('.ti')[0]+'_label.tif' #gid
#     tmp='div_train/images/'+file+'   '+'div_train/masks/'+file
#     target_file.write(tmp+'\n')

# source='/home/yf/disk/camera/test'
# txt='/home/yf/disk/camera/test.lst'
# filelist=os.listdir(source)
# target_file=open(txt,'w')
for file in filelist:
    #tmp='div_train/images/'+file+'   '+'div_train/masks/'+file.split('.ti')[0]+'_label.tif' #gid
    # tmp='test/'+file
    #tmp='div_train/images/'+file
    target_file.write(file.split('.')[0]+'\n')
# for lanelme
# for file in filelist:
#     #tmp='div_train/images/'+file+'   '+'div_train/masks/'+file.split('.ti')[0]+'_label.tif' #gid
#     tmp='div_train/images/'+file+'   '+'div_train/masks/'+file
#     target_file.write(tmp+'\n')