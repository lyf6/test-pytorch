import torch
from torch.nn import *
import torch.nn.functional as F

class mynet(Module):
    def __init__(self):
        super(mynet,self).__init__()
        self.layer1 = Conv2d(4, 4, kernel_size=1, bias=False)
        self.layer2 = Conv2d(4, 2, kernel_size=1, bias=False)
        self.layer3 = Sequential(
                Conv2d(2, 6, kernel_size=1, bias=False),
                Conv2d(6, 10, kernel_size=1, bias=False))
    def forward(self, input):
        out = self.layer1(input)
        fature = self.layer2(out)
        out = self.layer3(fature)
        return out, fature

class outmodel(Module):
    def __init__(self):
        super(outmodel,self).__init__()
        self.model=mynet()
    def forward(self, input):
        out, feature=self.model(input)
        return  out, feature


class nannanmodel(Module):
    def __init__(self):
        super(nannanmodel,self).__init__()
        self.model1=outmodel()
        self.conv = Conv2d(10, 12, kernel_size=1, bias=False)
    def forward(self, input):
        out, feature=self.model1(input)
        out = self.conv(out)
        return  out, feature


input=torch.ones(size=(5, 4, 8, 8))
model=nannanmodel()
#a=DataParallel(a)
out, feature = model(input)
dict_of_model = model.state_dict()

path= '/home/yf/disk/spacenet5/spacenet5/results/weights/sn5_baseline/test.pth'
torch.save(a,path)
feature = F.upsample(feature,(10,10), mode='bilinear')
print(feature.shape)
print(a)



