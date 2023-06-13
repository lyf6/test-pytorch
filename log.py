import os
import shutil
root='/home/yf/disk/lczlog'
listdirs=os.listdir(root)
target = '/home/yf/disk/onlycheckpoint'
for dir in listdirs:
    tmp = os.path.join(root,dir)
    sec = os.listdir(tmp)
    for d in sec:
        if 'checkpoint' in d:
            target_dir = os.path.join(target,dir)
            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)
            src = os.path.join(root,dir,d)
            for file in os.listdir(src):
                #file = os.path.join(src,file)
                if 'best' in file:
                    file = os.path.join(src,file)
                    shutil.copy(file,target_dir)
