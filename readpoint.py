from laspy.file import File
import numpy as np
import torch
# inFile = File('/home/yf/disk/pointcloud/57.las', mode='r')
# print('hhh')

a = torch.tensor([[1,2,3],
                  [4,5,6]])
b = a*a
print(a)
print(b)