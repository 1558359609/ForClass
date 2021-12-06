import math
import torch

x=torch.tensor(2.,requires_grad=True)

t=lambda x:x**3

x.grad=None
for i in range(3):

    y=t(x)
    y.backward()

    print(x.grad)
    x.grad=None