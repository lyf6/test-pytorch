import os


source='/home/yf/disk/sensetime/changedet/train/im1'
train='/home/yf/Documents/mmsegmentation/data/sensetime/train.txt'
val='/home/yf/Documents/mmsegmentation/data/sensetime/val.txt'
filelist=os.listdir(source)
train_file=open(train,'w')
test_file=open(val,'w')
ratio = 0.3
num_of_train = int(ratio*len(filelist))
cnt = 0
for file in filelist:
    if cnt<num_of_train:
        train_file.write('im1/'+file.split('.')[0]+'\n')
        train_file.write('im2/'+file.split('.')[0]+'\n')
        cnt+=1
    else:
        test_file.write('im1/'+file.split('.')[0]+'\n')
        test_file.write('im2/'+file.split('.')[0]+'\n')
        cnt+=1


# for file in filelist:
#     #tmp='div_train/images/'+file+'   '+'div_train/masks/'+file.split('.ti')[0]+'_label.tif' #gid
#     tmp='div_train/images/'+file+'   '+'div_train/masks/'+file
#     target_file.write(tmp+'\n')

# source='/home/yf/disk/camera/test'
# txt='/home/yf/disk/camera/test.lst'
# filelist=os.listdir(source)
# target_file=open(txt,'w')
# for file in filelist:
#     #tmp='div_train/images/'+file+'   '+'div_train/masks/'+file.split('.ti')[0]+'_label.tif' #gid
#     tmp='test/'+file
#     #tmp='div_train/images/'+file
#     target_file.write(tmp+'\n')
# for lanelme
# for file in filelist:
#     #tmp='div_train/images/'+file+'   '+'div_train/masks/'+file.split('.ti')[0]+'_label.tif' #gid
#     tmp='div_train/images/'+file+'   '+'div_train/masks/'+file
#     target_file.write(tmp+'\n')