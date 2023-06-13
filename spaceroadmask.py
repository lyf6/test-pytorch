import os

root='/home/yf/disk/spacenet2/SV_train/'
imagpath=root+'images/'
maskpath=root+'masks/'
txt='/home/yf/disk/spacenet2/train.lst'
filelist=os.listdir(imagpath)
train_file=open(txt,'w')
len_num=len(filelist)

for file in filelist:
    mask=file
    tmp=imagpath+file+'   '+ maskpath+mask
    train_file.write(tmp+'\n')
