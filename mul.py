import torch
import torch.nn.functional as F
import numpy as np
a=np.array([[1.0,2.0, 3.0, 4.0],
           [4.0,2.0,2.0,4.0]])
c=np.arrcy([])
a=torch.from_numpy(a)
b=F.softmax(a,dim=1)
print(a)
print(b)


