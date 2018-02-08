from __future__ import print_function
import torch
import numpy as np



x = torch.Tensor(5,3)
print(x)
x = torch.rand(5,3)
print(x)
print(x.size())

y = torch.rand(5,3)
print(x + y)
print(torch.add(x,y))
z = torch.Tensor(5,3)
torch.add(x,y,out=z)
print(z)

y.add_(x)
print(y)
print(y[:,1])

x = torch.randn(4,4)
y = x.view(16)       # reshape but in reference
z = x.view(-1,8)
print(x, y, z)

y[7] = 1000.0
print(x,y,z)

a = torch.ones(5)
print(a)
b = a.numpy()  #reference in numpy type
print(b)

a.add_(1)
print(a)
print(b)

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a,b)

if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    print(x+y)
    