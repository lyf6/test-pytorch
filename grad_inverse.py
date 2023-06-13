import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Function

class GradReverse(Function):
    def __init__(self, lamda):
        self.lamda=lamda

    def forward(self,input):
        return input

    def backward(self,grad_output):
        return (-self.lamda*grad_output)

x0=torch.tensor(1.2,requires_grad=True)
x1=torch.sigmoid(x0).requires_grad_()
x1.retain_grad()
x1z=GradReverse(2)(x1).requires_grad_()
x1z.retain_grad()
x2=x1z*x1z
x2.retain_grad()
loss=x2*x2
loss.backward()
print(x0.grad)
print(x1.grad)
print(x1z.grad)
print(x2.grad)