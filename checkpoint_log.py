import os
import shutil
root='/home/yf/disk/lczlog'
listdirs=os.listdir(root)
target = '/home/yf/disk/checkpoint_log'
target_file = 'exp.txt'
myFile = open(target_file)


for file in myFile.readlines():
    file = file.split('\n')[0]
    print(file)
    hitfile = list(filter(lambda x: file in x, listdirs))
    print(hitfile)
    for dir in hitfile:
        #print(dir)
        tmp = os.path.join(root,dir)
        sec = os.listdir(tmp)
        for d in sec:
            if 'checkpoint' in d:
                target_dir = os.path.join(target,dir)
                if not os.path.isdir(target_dir):
                    os.makedirs(target_dir)
                src = os.path.join(root,dir,d)
                #print(src)
                for file in os.listdir(src):
                    #file = os.path.join(src,file)
                    if 'best' in file:
                        tmp = file
                        file = os.path.join(src,file)

                        if os.path.exists(file):
                            if not os.path.exists(os.path.join(target_dir, tmp)):
                                shutil.copy(file,target_dir)
                        #print(file)
            if 'log' in d:
                target_dir = os.path.join(target,dir)
                if not os.path.isdir(target_dir):
                    os.makedirs(target_dir)
                src = os.path.join(root,dir,d)
                for file in os.listdir(src):
                    #file = os.path.join(src,file)
                    file = os.path.join(src,file)
                    #shutil.copy(file,target_dir)

