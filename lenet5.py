import torch
from torch.nn import *
from torchvision import datasets, transforms
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import argparse

class lenet5(Module):
    def __init__(self):
        super(lenet5, self).__init__()
        self.layer1 = Conv2d(1, 6, kernel_size=5, bias=False)  # 28
        self.layer2 = MaxPool2d(kernel_size=2, stride=2)  # 14
        self.layer3 = Conv2d(6, 16, kernel_size=5, bias=False)  # 10
        self.layer4 = MaxPool2d(kernel_size=2, stride=2)  # 5
        self.layer5 = Linear(400, 120)  # 5*5*16
        self.layer6 = Linear(120, 84)
        self.layer7 = Linear(84, 10)

    def forward(self, input):
        feature = self.layer1(input)
        feature = F.relu(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        feature = F.relu(feature)
        feature = self.layer4(feature)
        feature = feature.view(-1, 400)
        feature = self.layer5(feature)
        feature = F.relu(feature)
        feature = self.layer6(feature)
        feature = F.relu(feature)
        # feature = F.softmax(feature)
        log = self.layer7(feature)
        return log, feature

parser = argparse.ArgumentParser(description='lenet5')
parser.add_argument('--batch_size', type=int, default=100,
                    help='number of input samples '
                         'to train (default: 256)')
parser.add_argument('--num_workers', type=int, default=6,
                    help='number of workers '
                         'to load data (default: 6)')
args = parser.parse_args()

transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

init_lr = 0.01
num_epoch = 5
semi = True

data_train = datasets.MNIST(root = "./data/",
                            transform=transform,
                            train = True,
                            download = True)

data_test = datasets.MNIST(root="./data/",
                           transform = transform,
                           train = False)

train_loder = DataLoader(data_train,
                          shuffle=True,
                          batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
unlabeled_loder = DataLoader(data_test,
                          shuffle=True,
                          batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
test_loder = DataLoader(data_test,
                          shuffle=True,
                          batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)


model = lenet5()
opt = SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.0005)

def cross_entropy(pred, gt, weight=1.0):
    log_pred = (pred+1e-8).log()
    loss=(log_pred * gt).sum() * weight * (-1)/gt.shape[0]
    return loss

def addsemiLoss(a, b, labels):
    m = torch.mm(a, b.T)
    p_ab = F.softmax(m, 1)
    p_ba = F.softmax(m.T, 1)
    p_aba = torch.mm(p_ab, p_ba)
    #label_withoutOnehot = torch.argmax(labels, dim=1)
    T_walker_eq = torch.eq(labels, labels.view(-1,1)).float()
    T_walker = T_walker_eq / T_walker_eq.sum(1, keepdim=True)
        #F.softmax(T_walker_eq, 1)
    Loss_Walker = cross_entropy(p_aba, T_walker)*1.0

    T_visit = torch.full((b.shape[0],1), 1.0/b.shape[0])
    p_visit_sum = p_ab.mean(dim=0)
    Loss_Visit = cross_entropy(p_visit_sum, T_visit)*0.2

    return Loss_Walker, Loss_Visit

def train(train_loder):
   model.train()
   for batchid, data in enumerate(train_loder):
       opt.zero_grad()
       train_x, y_true = data
       log_x, fea_x= model(train_x.float())
       #y_true = F.one_hot(y_true)
       #softmax_log_x = F.softmax(log_x, dim=1)
       #total_loss = cross_entropy(softmax_log_x, y_true)
       total_loss = F.cross_entropy(log_x, y_true)

       if semi:
          try :
              ul_x, ul_y = next(unlabeled_loder)
          except TypeError:
              unlabeled = iter(unlabeled_loder)
              ul_x, ul_y = next(unlabeled)

          ul_x = ul_x.float()
          log_ul_x, fea_ul_x = model(ul_x)

          loss_walker, loss_visit = addsemiLoss(fea_x, fea_ul_x, y_true)
          print('classification loss is %f' % total_loss)
          total_loss = total_loss + loss_walker + loss_visit
          print('walker loss is %f' % loss_walker)
          print('visit loss is %f' % loss_visit)
          print('total loss is %f' % total_loss)

       total_loss.backward()
       opt.step()

   #return classification_loss

def add_visit_loss(p, visit_weight=1.0):
    visit_probability = p.mean(dim=0)
    t_nb = p.shape[1]
    visit_target = torch.full((1, t_nb), 1.0/t_nb)
    visit_loss = cross_entropy(visit_probability,visit_target, weight=visit_weight)
    return visit_loss

def add_semisup_loss(a, b, labels, walker_weight=1.0, visit_weight=1.0):

    a_unit = a/torch.norm(a,p=2,dim=1, keepdim=True)
    b_unit = b/torch.norm(b,p=2,dim=1, keepdim=True)
    cosin_ab = torch.mm(a_unit, b_unit.transpose(1,0))*10

    match_ab = torch.mm(a, b.transpose(1,0))
    p_ab = F.softmax(match_ab, 1)
    p_ba = F.softmax(match_ab.transpose(0,1),1)
    p_aba = torch.mm(p_ab, p_ba)
    equality_matrix = torch.eq(labels, labels.view(-1,1)).float()
    p_target = equality_matrix / equality_matrix.sum(dim=1, keepdim=True)
    walk_loss = cross_entropy(p_aba, p_target, weight=walker_weight)
    visit_loss = add_visit_loss(p_ab, visit_weight)
    semi_loss = walk_loss+visit_loss
    return semi_loss,walk_loss, visit_loss

def Reftrain(train_loder):
   model.train()
   for batchid, data in enumerate(train_loder):
       opt.zero_grad()
       train_x, y_true = data
       log_x, fea_x= model(train_x.float())
       #y_true = F.one_hot(y_true)
       total_loss = F.cross_entropy(log_x, y_true)
       print('classification loss is %f' % total_loss)
       if semi:
          try :
              ul_x, ul_y = next(unlabeled_loder)
          except TypeError:
              unlabeled = iter(unlabeled_loder)
              ul_x, ul_y = next(unlabeled)

          ul_x = ul_x.float()
          log_ul_x, fea_ul_x = model(ul_x)

          semi_loss, loss_walker, loss_visit = add_semisup_loss(fea_x, fea_ul_x, y_true)
          total_loss = total_loss + semi_loss
          print('walker loss is %f' % loss_walker)
          print('visit loss is %f' % loss_visit)
          print('total loss is %f' % total_loss)

       total_loss.backward()
       opt.step()

def eval(test_loder):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batchid, data in enumerate(test_loder):
            test_x, y_true = data
            log_x, _ = model(test_x)
            pred = np.argmax(log_x, axis=1)
            correct += (pred == y_true).sum()
            total += y_true.size(0)
        acc = 100.*correct/total
    print("acc is %f"%acc)

def main():

    for e in range(num_epoch):
        Reftrain(train_loder)
        # train(train_loder)
        eval(test_loder)


if __name__ == '__main__':
    main()