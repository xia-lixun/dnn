import numpy as np
import scipy.io as scio
import os, os.path




def dataset_dimension(path):

    n_parts = len([name for name in os.listdir(path)])
    spect_examples = 0
    label_examples = 0
    spect_width = 0
    label_width = 0

    for p in range(1, 1+n_parts):
        temp = scio.loadmat(os.path.join(path, 't_' + str(p) + '.mat'))
        spect_examples += temp['spec'].shape[0]
        label_examples += temp['bm'].shape[0]
        spect_width = temp['spec'].shape[1]
        label_width = temp['bm'].shape[1]
        del temp
        print('populating: %d/%d' % (p, n_parts))

    return spect_examples, spect_width, label_examples, label_width, n_parts




def dataset_load2mem(path):

    m, n, p, q, n_parts = dataset_dimension(path)
    spect = np.zeros((m,n), dtype='float32')
    label = np.zeros((p,q), dtype='float32')
    offset = 0

    for i in range(1, 1+n_parts):
        temp = scio.loadmat(os.path.join(path, 't_' + str(i) + '.mat'))
        instance = temp['spec'].shape[0]
        spect[offset:offset+instance] = temp['spec']
        label[offset:offset+instance] = temp['bm']
        offset += instance
        del temp
        print('loading: %d/%d'%(i,n_parts))

    return spect, label



x,y= dataset_load2mem('/home/coc/6-Workspace/test')
print('total dimension: ')