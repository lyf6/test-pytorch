import os
from PIL import Image
import numpy as np
dir='/home/yf/Documents/py3.8/mmsegmentation/data/tianchi/ann_dir'
#'/home/yf/Documents/py3.8/mmsegmentation/data/tianchi/ann_dir'
files=os.listdir(dir)
classnum=11
# count={0:0, 1:0, 2:0, 3:0, 4:0}
#classnum=12
count={id:0.0 for id in range(classnum)}
for file in files:
    path = os.path.join(dir, file)
    img = np.array(Image.open(path))
    for labeled in count:
        sum=(img==labeled).sum()
        count[labeled]+=sum
print(count)

sum=0
for labeled in count:
    count[labeled]=1.0/np.log(count[labeled]+1)
#    sum+=count[labeled]
# print(sum)
#for labeled in count:
    #count[labeled]=classnum*count[labeled]/sum
#   count[labeled] = count[labeled]/sum
    #count[labeled]=(sum/count[labeled]*0.01)


print(count)
