import os
import random



target='/home/yf/Documents/mmclassification/data/google/'
source=target+'data'



cls_dir=os.listdir(source)
expnum = 5
train_ratio=0.8 #need change
# for file in filelist:
#     #tmp='div_train/images/'+file+'   '+'div_train/masks/'+file.split('.ti')[0]+'_label.tif' #gid
#     tmp='div_train/images/'+file+'   '+'div_train/masks/'+file
#     target_file.write(tmp+'\n')

# source='/home/yf/disk/camera/test'
# txt='/home/yf/disk/camera/test.lst'
# filelist=os.listdir(source)
# target_file=open(txt,'w')
for expid in range(expnum):
    train_file = target+'train_'+str(expid)+'.lst'
    test_file = target+'test_'+str(expid)+'.lst'
    train = open(train_file,'w')
    test = open(test_file,'w')
    for cls_id in range(len(cls_dir)):
        cls_path = os.path.join(source, cls_dir[cls_id])     
        img_num = len(os.listdir(cls_path))
        train_num = int(img_num*train_ratio)
        test_num = img_num - train_num
        print(cls_dir[cls_id], img_num, train_num)
        cnt = 0
        img_list = os.listdir(cls_path)
        random.shuffle( img_list )
        for img in img_list:
            tmp = cls_dir[cls_id]+'/'+img + ' ' + str(cls_id)
            #print(tmp)
            
            if cnt<train_num:
                train.write(tmp+'\n')
            else:
                test.write(tmp+'\n')
            cnt = cnt+1
        