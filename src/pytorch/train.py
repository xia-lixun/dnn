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

import numpy as np
import scipy.io as scio
import h5py
import time
import os, os.path
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split




path_train = '/home/coc/workspace/train/'
path_test = '/home/coc/workspace/test/'
path_model = '/home/coc/workspace/model-20180214.mat'



def loadmat_transpose(path):
    temp = {}
    f = h5py.File(path)
    for k,v in f.items():
        temp[k] = np.array(v)
    return temp

def dataset_size(path):
    num_files = len(os.listdir(path))
    total_bytes = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))
    return num_files, total_bytes



# tensor size and system memory utilization
mem_util_ratio = 0.7
mem_available = psutil.virtual_memory().available

test_files, test_bytes = dataset_size(path_test)
train_files, train_bytes = dataset_size(path_train)

test_partitions = int(test_bytes // (mem_available * mem_util_ratio)) + 1
train_partitions = int(train_bytes // (mem_available * mem_util_ratio * 0.5)) + 1
print('[init]: train, %f MiB in %d partitions'%(train_bytes/1024/1024, train_partitions))
print('[init]: test, %f MiB in %d partitions'%(test_bytes/1024/1024, test_partitions))

# GPU memory utillization
gpu_mem_available = 11*1024*1024*1024
gpu_mem_util_ratio = 0.4
test_batch_partitions = int((test_bytes/test_partitions) // (gpu_mem_available * gpu_mem_util_ratio)) + 1
print('[init]: test, %d GPU partitions for each CPU partition'%(test_batch_partitions))

# find out the tensor dimensions
tensor = loadmat_transpose(os.path.join(path_train, 't_1.mat'))
input_dim = tensor['variable'].shape[1]
output_dim = tensor['label'].shape[1]
del tensor
print('[init]: input,output dims = %d,%d' % (input_dim, output_dim))



n_epochs = 400
batch_size_init = 128
learn_rate_init = 0.01
dropout_prob = 0.3
momentum_coeff = 0.9
rng = np.random.RandomState(4913)







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
        x = F.dropout(x, p=dropout_prob, training=self.training)
        x = F.sigmoid(self.fc2(x))
        x = F.dropout(x, p=dropout_prob, training=self.training)
        x = F.sigmoid(self.fc3(x))
        x = F.dropout(x, p=dropout_prob, training=self.training)
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
optimizer = optim.SGD(model.parameters(), lr=learn_rate_init, momentum=momentum_coeff)
criterion = nn.MultiLabelSoftMarginLoss(size_average=False)






def indexing_dimension(path):
    dim = []
    dim.append((0,0,0,0))

    for i in range(1, 1+len([x for x in os.listdir(path)])):
        temp = loadmat_transpose(os.path.join(path, 't_' + str(i) + '.mat'))
        var_samples = temp['variable'].shape[0]
        var_width = temp['variable'].shape[1]
        lab_samples = temp['label'].shape[0]
        lab_width = temp['label'].shape[1]
        assert(var_samples == lab_samples)
        dim.append((var_samples,var_width, lab_samples, lab_width))
    return dim



def dataset_dimension(index, select):
    #note: select can be in any order! [7, 3, 9, 11, 4, ...]
    var_samples = 0
    var_width = index[1][1]
    lab_samples = 0
    lab_width = index[1][3]

    for i in select:
        var_samples += index[i][0]
        lab_samples += index[i][2]
    return var_samples, var_width, lab_samples, lab_width



def dataset_load2mem(path, index, select):
    #note: select can be in any order! [7, 3, 9, 11, 4, ...]
    m, n, p, q = dataset_dimension(index, select)

    variable = np.zeros((m,n), dtype='float32')
    label = np.zeros((p,q), dtype='float32')
    offset = 0

    for i in select:
        #temp = scio.loadmat(os.path.join(path, 't_' + str(i) + '.mat'))
        temp = loadmat_transpose(os.path.join(path, 't_' + str(i) + '.mat'))
        stride = temp['variable'].shape[0]
        variable[offset:offset+stride] = temp['variable']
        label[offset:offset+stride] = temp['label']
        offset += stride
        del temp
    return variable, label



def evaluate_batch_cost(variable, label):
    loss = 0.0
    for i in np.array_split(range(label.shape[0]), test_batch_partitions):

        data = torch.from_numpy(variable[i[0]:i[-1]])
        target = torch.from_numpy(label[i[0]:i[-1]])

        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        output = model(data)
        loss += criterion(output, target).data[0]
    return loss



def evaluate_total_cost(index):
    loss = 0.0
    examples = 0
    for portion in np.array_split(range(1,1+len([x for x in os.listdir(path_test)])), test_partitions):
        test_spect, test_label = dataset_load2mem(path_test, index, portion)
        # print('spectrum size %d %d' % (test_spect.shape[0], test_spect.shape[1]))
        # print('label size %d %d' %(test_label.shape[0], test_label.shape[1]))
        loss += evaluate_batch_cost(test_spect, test_label)
        examples += test_label.shape[0]
        del test_spect
        del test_label
    print('[info]: total examples for eval %d'%(examples))
    return loss/examples



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