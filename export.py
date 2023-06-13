import torch
import torch.nn as nn

class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
        self.a = torch.tensor(5)
        self.b = torch.tensor(6)

    def forward(self):
        return self.a + self.b

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

dummy_input = torch.randn(10, 3, 224, 224)
model = test()
result = model()
print(result)
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(model, "test.onnx", verbose=True, input_names=input_names, output_names=output_names)
