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
import psutil
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split




path_train = '/home/coc/Public/train/tensor/'
path_test = '/home/coc/Public/test/tensor/'
path_model = '/home/coc/Public/model-pytorch.mat'




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
gpu_mem_util_ratio = 0.2
test_batch_partitions = int((test_bytes/test_partitions) // (gpu_mem_available * gpu_mem_util_ratio)) + 1
print('[init]: test, %d GPU partitions for each CPU partition'%(test_batch_partitions))

# find out the tensor dimensions
tensor = loadmat_transpose(os.path.join(path_train, 't_1.mat'))
input_dim = tensor['variable'].shape[1]
output_dim = tensor['label'].shape[1]
del tensor
print('[init]: input,output dims = %d,%d' % (input_dim, output_dim))
hidden_dim = 2048




n_epochs = 400
batch_size_init = 128
learn_rate_init = 0.01
#dropout_prob = 0.3
momentum_coeff = 0.9
torch.manual_seed(65537)

t_init = time.time()
dim_test = indexing_dimension(path_test)
dim_train = indexing_dimension(path_train)
t_stop = time.time()
print('[init]: dimension indexing done, %.3f (sec)'%(t_stop-t_init))







class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

        # self.dp1 = nn.Dropout(p=dropout_prob)
        # self.dp2 = nn.Dropout(p=dropout_prob)
        # self.dp3 = nn.Dropout(p=dropout_prob)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim, momentum=0.01) # eps = 1e-5,affine=True,track_running_stats=True
        self.bn2 = nn.BatchNorm1d(hidden_dim, momentum=0.01)
        self.bn3 = nn.BatchNorm1d(hidden_dim, momentum=0.01)
        self.bn4 = nn.BatchNorm1d(output_dim, momentum=0.01)

        nn.init.xavier_normal(self.fc1.weight)
        nn.init.xavier_normal(self.fc2.weight)
        nn.init.xavier_normal(self.fc3.weight)
        nn.init.xavier_normal(self.fc4.weight)
        nn.init.constant(self.fc1.bias, 0.0)
        nn.init.constant(self.fc2.bias, 0.0)
        nn.init.constant(self.fc3.bias, 0.0)
        nn.init.constant(self.fc4.bias, 0.0)

    def forward(self, x):
        x = F.sigmoid(self.bn1(self.fc1(x)))
        # x = self.dp1(x)
        x = F.sigmoid(self.bn2(self.fc2(x)))
        # x = self.dp2(x)
        x = F.sigmoid(self.bn3(self.fc3(x)))
        # x = self.dp3(x)
        x = self.bn4(self.fc4(x))
        return x


model = Net()
model.cuda()
params = list(model.parameters())
print(model)
# print(len(params))
# print(params[0].size())



# [placeholder]: paramter init block...
optimizer = optim.SGD(model.parameters(), lr=learn_rate_init, momentum=momentum_coeff)
criterion = nn.MultiLabelSoftMarginLoss()









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


def evaluate_batchcost(variable, label):
    loss = 0.0
    for i in np.array_split(range(label.shape[0]), test_batch_partitions):

        data = torch.from_numpy(variable[i[0]:i[-1]])
        target = torch.from_numpy(label[i[0]:i[-1]])
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        output = model(data)
        loss += criterion(output, target).item()
    return loss/test_batch_partitions



def evaluate_testcost(index):
    loss = 0.0
    for portion in np.array_split(range(1,1+len([x for x in os.listdir(path_test)])), test_partitions):
        spectrum, label = dataset_load2mem(path_test, index, portion)
        # print('spectrum size %d %d' % (spectrum.shape[0], spectrum.shape[1]))
        # print('label size %d %d' %(label.shape[0], label.shape[1]))
        loss += evaluate_batchcost(spectrum, label)
        del spectrum
        del label
    return loss/test_partitions



def train(epoch, batchsize):
    model.train()
    loss_a = 0.0

    for rand_portion in np.array_split(shuffle(range(1,1+len([x for x in os.listdir(path_train)]))), train_partitions):

        spectrum, label = dataset_load2mem(path_train, dim_train, rand_portion)
        spectrum, label = shuffle(spectrum, label)
        n_batches = label.shape[0] // batchsize
        loss_b = 0.0

        for i in range(n_batches):
            a = i * batchsize
            b = a + batchsize

            data, target = torch.from_numpy(spectrum[a:b]), torch.from_numpy(label[a:b])
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()   
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss_b += loss.item()

        loss_a += (loss_b / n_batches)
        del spectrum
        del label

    return loss_a/train_partitions



def test():
    model.eval()
    with torch.no_grad():
        loss = evaluate_testcost(dim_test)
    return loss
    





test_loss_min = 1000000.0
test_loss = test()
print('[init]: validation loss = %.3f ' % (test_loss))

for epoch in range(1,1+n_epochs):
    t_init = time.time()
    train_loss = train(epoch, batch_size_init)
    test_loss = test()
    t_stop = time.time()
    print('------------------------------------------------------------')
    print('[epoch %i] loss = %.3f,%.3f'%(epoch, train_loss, test_loss))
    print('[epoch %i] time = %.3f [sec]'%(epoch, t_stop-t_init))

    if (test_loss < test_loss_min):
        model_dict = {}
        model_dict['W1'] = model.fc1.weight.data.cpu().numpy()
        model_dict['W2'] = model.fc2.weight.data.cpu().numpy()
        model_dict['W3'] = model.fc3.weight.data.cpu().numpy()
        model_dict['W4'] = model.fc4.weight.data.cpu().numpy()

        model_dict['b1'] = model.fc1.bias.data.cpu().numpy()
        model_dict['b2'] = model.fc2.bias.data.cpu().numpy()
        model_dict['b3'] = model.fc3.bias.data.cpu().numpy()
        model_dict['b4'] = model.fc4.bias.data.cpu().numpy()

        scio.savemat(path_model, model_dict)
        test_loss_min = test_loss
        print('[epoch %d] ||||||||||||||||||||||||||||||||||||||||||||||||||' % (epoch))






# https://github.com/pytorch/examples/blob/master/mnist/main.py

# Is it the gradient of the eventual downstream loss with respect to the current layer? 
# So that in the case of a scalar loss which is also the “most downstream output/loss” 
# we get dloss/dloss =1 but if we want to get backward() from some middle layer we have 
# to provide the gradient of the downstream loss w.r.t. all the outputs of this middle 
# layer (evaluated at the current values of those outputs) in order to get well defined 
# numerical results. This makes sense to me and actually occurs in backprop.
# In more technical terms. Let y be an arbitrary node in a computational graph If we call 
# y.backward(arg) the argument arg to backward should be the gradient of the root of the 
# computational graph with respect to y evaluated at a specific value of y 
# (usually the current value of y). If y is a whole layer, this means that arg should 
# provide a value for each neuron in y. If y is th final loss it is also the root of the 
# graph and we get the usual scalar one as the only reasonable argument arg.
#
# Yes, that’s correct. We only support differentiation of scalar functions, so if you want 
# to start backward form a non-scalar value you need to provide dout / dy