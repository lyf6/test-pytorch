import os


source='/home/yf/disk/testdata/2019jn'
txt='/home/yf/disk/SJD/2019_test.lst'
filelist=os.listdir(source)
target_file=open(txt,'a')
for file in filelist:
    #tmp='div_train/images/'+file+'   '+'div_train/masks/'+file.split('.ti')[0]+'_label.tif' #gid
    tmp = 'test/2019jn/' + file
        #'div_val/images/'+file+'   '+'div_val/masks/'+file
    target_file.write(tmp+'\n')