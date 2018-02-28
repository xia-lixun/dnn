# -*- coding: utf-8 -*-
"""
Neural Networks
===============

Neural networks can be constructed using the ``torch.nn`` package.

Now that you had a glimpse of ``autograd``, ``nn`` depends on
``autograd`` to define models and differentiate them.
An ``nn.Module`` contains layers, and a method ``forward(input)``\ that
returns the ``output``.

For example, look at this network that classfies digit images:

.. figure:: /_static/img/mnist.png
   :alt: convnet

   convnet

It is a simple feed-forward network. It takes the input, feeds it
through several layers one after the other, and then finally gives the
output.

A typical training procedure for a neural network is as follows:

- Define the neural network that has some learnable parameters (or
  weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network’s parameters
- Update the weights of the network, typically using a simple update rule:
  ``weight = weight - learning_rate * gradient``

Define the network
------------------

Let’s define this network:
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear((31+1) * 136, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 136)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.sigmoid(self.fc2(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.sigmoid(self.fc3(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc4(x)
        return x


model = Net()
model.cuda()

params = list(model.parameters())
print(model)
print(len(params))
print(params[0].size())
print(params[0])



# [placeholder]: paramter init block...
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.MultiLabelSoftMarginLoss(size_average=False)



def train(epoch):
    model.train()


def test():
    model.eval()
    





# [placeholder]: init cost of the training and validation set
input = Variable(torch.randn(100, (31+1) * 136))
target = Variable(torch.randn(100, 136))

output = model(input)
loss = criterion(output, target)
print(loss)
print(loss.grad_fn) 
print(loss.grad_fn.next_functions[0][0])  
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) 






# for epoch in range(1,1+100):

#     optimizer.zero_grad()   # zero the gradient buffers
#     output = model(input)
#     loss = criterion(output, target)
#     print(model.fc1.bias.grad)
#     loss.backward()
#     print(model.fc1.bias.grad)
#     optimizer.step()    # Does the update


https://github.com/pytorch/examples/blob/master/mnist/main.py