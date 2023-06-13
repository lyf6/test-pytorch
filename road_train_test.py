import os

root='/home/yf/disk/spacenet2/'
imagpath=root+'8bit_ps/'
maskpath=root+'graymasks/'
train_txt='/home/yf/disk/spacenet2/train.lst'
test_txt='/home/yf/disk/spacenet2/val.lst'
filelist=os.listdir(imagpath)
train_file=open(train_txt,'w')
test_file=open(test_txt, 'w')
len_num=len(filelist)
count=0
train_percent=0.8
trainnum=int(len_num*train_percent)
for file in filelist:
    if count<trainnum:
        tmp=imagpath+file+'   '+ maskpath+file
        train_file.write(tmp+'\n')
    else:
        tmp = imagpath + file + '   ' + maskpath + file
        test_file.write(tmp+'\n')
    count=count+1