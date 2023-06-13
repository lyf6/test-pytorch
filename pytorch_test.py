import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
torch.distributed.init_process_group(backend="nccl")

batch_size = 1
data_size = 8

local_rank = torch.distributed.get_rank()

torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

class RandomDataset(Dataset):
        def __init__(self, length):
            self.len = length
            self.data = torch.stack([torch.ones(1), torch.ones(1)*2,torch.ones(1)*3,torch.ones(1)*4,torch.ones(1)*5,torch.ones(1)*6,torch.ones(1)*7,torch.ones(1)*8]).to('cuda')

        def __getitem__(self, index):
            return self.data[index]
        def __len__(self):
            return self.len

dataset = RandomDataset(data_size)
sampler = DistributedSampler(dataset)

#rand_loader =DataLoader(dataset=dataset,batch_size=batch_size,sampler=None,shuffle=True)
rand_loader = DataLoader(dataset=dataset,batch_size=batch_size,sampler=sampler)
epoch = 0
while epoch < 4:
    #print('epoch: '+ str(epoch))

    for data in rand_loader:
            print(len(rand_loader))
            print(data, epoch)
    epoch+=1
