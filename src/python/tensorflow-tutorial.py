import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import scipy.io as scio
import h5py
import time




##########################
#########MODEL##########
##########################
rng = np.random.RandomState(1234)


def make_window_buffer(x, neighbor=3):
    m, n = x.shape
    tmp = np.zeros(m * n * (neighbor * 2 + 1), dtype='float32').reshape(m, -1)
    for i in range(2 * neighbor + 1):
        if (i <= neighbor):
            shift = neighbor - i
            tmp[shift:m, i * n: (i + 1) * n] = x[:m - shift]
            for j in range(shift):
                tmp[j, i * n: (i + 1) * n] = x[0, :]
        else:
            shift = i - neighbor
            tmp[:m-shift, i * n: (i+1) * n] = x[shift:m]
            for j in range(shift):
                tmp[m-(j + 1), i * n: (i + 1) * n] = x[m-1, :]
    return tmp

def Normalize_data(x, mu, std):
    mean_noisy_10 = np.tile(mu, [8])
    std_noisy_10 = np.tile(std, [8])
    tmp = (x-mean_noisy_10)/std_noisy_10
    return np.array(tmp, dtype='float32')

def Normalize_label(x, mu, std):
    tmp = (x-mu)/std
    return np.array(tmp, dtype='float32')

def gen_context(x, neighbor, gmu, gstd):
    m = x.shape[0]
    u = make_window_buffer(x, neighbor)
    nat = np.zeros([m, 257])
    for k in range(0,7):
        nat += u[:, k*257:(k+1)*257]
    u = np.c_[u, nat/7]
    u = Normalize_data(u, gmu, gstd)
    return u
# u: np.zeros([m, 257*8])




class Autoencoder:

    def __init__(self, vis_dim, hid_dim, W, function=lambda x: x):
        self.W = W
        self.a = tf.Variable(np.zeros(vis_dim).astype('float32'), name='a')
        self.b = tf.Variable(np.zeros(hid_dim).astype('float32'), name='b')
        self.function = function
        self.params = [self.W, self.a, self.b]

    def encode(self, x):
        u = tf.matmul(x, self.W) + self.b
        return self.function(u)

    def decode(self, x):
        u = tf.matmul(x, tf.transpose(self.W)) + self.a
        return self.function(u)

    def f_prop(self, x):
        y = self.encode(x)
        return self.decode(y)

    def reconst_error(self, x, noise):
        tilde_x = x * noise
        reconst_x = self.f_prop(tilde_x)
        error = tf.reduce_mean(tf.reduce_sum((x - reconst_x)**2, 1))
        return error, reconst_x


class Dense:

    def __init__(self, in_dim, out_dim, function=lambda x: x):
        self.W = tf.Variable(rng.uniform(low = -0.1, high = 0.1, size=(in_dim, out_dim)).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
        self.function = function
        self.params = [self.W, self.b]
        self.ae = Autoencoder(in_dim, out_dim, self.W, self.function)

    def f_prop(self, x):
        u = tf.matmul(x, self.W) + self.b
        self.z = self.function(u)
        return self.z

    def pretrain(self, x, noise):
        cost, reconst_x = self.ae.reconst_error(x, noise)
        return cost, reconst_x


layers = [
    Dense(257*12, 2048, tf.nn.sigmoid),
    Dense(2048, 2048, tf.nn.sigmoid),
    Dense(2048, 2048, tf.nn.sigmoid),
    Dense(2048, 257)
]
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, 257*12])
t = tf.placeholder(tf.float32, [None, 257])

def f_props(layers, x):
    for i, layer in enumerate(layers):
        x = layer.f_prop(x)
        if(i != len(layers)-1):
            x = tf.nn.dropout(x, keep_prob)
    return x
y = f_props(layers, x)

cost_fine = tf.reduce_mean(tf.reduce_sum((y - t)**2, 1))
lrate_p = tf.placeholder(tf.float32)
mt_p = tf.placeholder(tf.float32)
train_fine = tf.train.MomentumOptimizer(learning_rate=lrate_p, momentum=mt_p).minimize(cost_fine)
saver = tf.train.Saver()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


n_epochs = 50
batch_size = 128
part_num_train = 10
part_num_valid = 10


##########################
#########DATA#############
##########################





##########################
#########PROCESS##########
##########################
print("FineTuning begin")
cost_valid_best = 1000000


Cost_validation = 0.0
for part_num_ in range(part_num_valid):
    # note: in h5 data are in N x 257/2056 format	
    #       transpose to maintain the format unchanged for tf
    fid_valid = h5py.File('/home/coc/4-Workspace/valid/tensor-' + str(part_num_) + '.h5')
    t_valid_data = np.array(fid_valid["/data"], dtype='float32')
    t_valid_label = np.array(fid_valid["/label"], dtype='float32')
    del fid_valid
    
    Cost_validation = Cost_validation + sess.run(cost_fine, feed_dict={x: t_valid_data, t: t_valid_label, keep_prob: 1.0})
    del t_valid_data
    del t_valid_label
Cost_validation = Cost_validation / part_num_valid
print('EPOCH: 0, Validation cost: %.3f ' % (Cost_validation))




# training starts here
for epoch in range(n_epochs):
    mt = 0.9
    lrate = 0.001
    #if(epoch>3):
    #    lrate = 0.0005
    #if(epoch>10):
    #    lrate = 0.0002
    # if(epoch>20):
    #    lrate = 0.0001
    if(epoch>10):
        lrate = 0.0005


    time_start = time.time()
    part_num_list = shuffle(range(part_num_train))

    for part_num in part_num_list:
        try:
            del data_part
            del _data
            del _label
        except:
            pass

        data_part = h5py.File('/home/coc/4-Workspace/train/tensor-' + str(part_num) + '.h5')
        _data = np.array(data_part["/data"], dtype='float32')
        _label = np.array(data_part["/label"], dtype='float32')
        del data_part

        _data, _label = shuffle(_data, _label)
        n_batches = _data.shape[0] // batch_size

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            sess.run(train_fine, feed_dict={x: _data[start:end], t: _label[start:end], keep_prob: 0.8, lrate_p: lrate, mt_p: mt})
        print('part %i finished'%(part_num))

    #Cost_validation = sess.run(cost_fine, feed_dict={x: t_valid_data, t: t_valid_label, keep_prob: 1.0})
    Cost_validation = 0.0
    for part_num_ in range(part_num_valid):
        # note: in h5 data are in N x 257/2056 format	
        #       transpose to maintain the format unchanged for tf
        fid_valid = h5py.File('/home/coc/4-Workspace/valid/tensor-' + str(part_num_) + '.h5')
        t_valid_data = np.array(fid_valid["/data"], dtype='float32')
        t_valid_label = np.array(fid_valid["/label"], dtype='float32')
        del fid_valid
    
        Cost_validation = Cost_validation + sess.run(cost_fine, feed_dict={x: t_valid_data, t: t_valid_label, keep_prob: 1.0})
        del t_valid_data
        del t_valid_label
    Cost_validation = Cost_validation / part_num_valid

    time_end = time.time()
    print('EPOCH: %i, Validation cost: %.3f ' % (epoch + 1, Cost_validation))
    print('Elapsed time for one epoch is %.3f' % (time_end - time_start))
    #LOG.write('EPOCH: %i, Validation cost: %.3f \n' %(epoch + 1, Cost_validation))
    #LOG.flush()

    if (Cost_validation < cost_valid_best):
        save_dict = {}
        save_dict['W1'] = sess.run(layers[0].W)
        save_dict['b1'] = sess.run(layers[0].b)
        save_dict['W2'] = sess.run(layers[1].W)
        save_dict['b2'] = sess.run(layers[1].b)
        save_dict['W3'] = sess.run(layers[2].W)
        save_dict['b3'] = sess.run(layers[2].b)
        save_dict['W4'] = sess.run(layers[3].W)
        save_dict['b4'] = sess.run(layers[3].b)

        MATFILE = '/home/coc/4-Workspace/train/specification-2017-11-13T16-50-41-801.mat'
        scio.savemat(MATFILE, save_dict)
        cost_valid_best = Cost_validation
        print('Model in EPOCH:%d is saved' % (epoch + 1))
        #LOG.write('Model in EPOCH:%d is saved' % (epoch + 1))
    saver.save(sess, '/home/coc/4-Workspace/train/specification-2017-11-13T16-50-41-801.tf')

#LOG.close()
del _data
del _label
sess.close()













# stat = h5py.File("D:\\4-Workspace\\mix\\train\\global.h5")
# mu = np.array(stat["/mu"], dtype='float32')
# std = np.array(stat["/std"], dtype='float32')
# n_train = np.int64(stat["/frames"])
#
#
# train = h5py.File("D:\\4-Workspace\\mix\\train\\train.h5", "r")
# _data = np.zeros([1, 257*8])
# _label = np.zeros([1, 257])
# for g in train:
#     mix = np.array(train[g + "/mix"], dtype='float32').transpose()
#     _data = np.concatenate((_data, gen_context(mix, 3, mu, std)), axis=0)
#
#     speech = np.array(train[g + "/speech"], dtype='float32').transpose()
#     _label = np.concatenate((_label, (speech-mu)/std), axis=0)
#
# _data = _data[1:_data.shape[0]]
# _label = _label[1:_data.shape[0]]
#
# del train


# x = np.array(valid["D:\\4-Workspace\\mix\\valid\\wav\\628+stationary+pink2688000+dr2+mbjk0+si2128+1+1+-22.0+5.0.wav/mix"], dtype='float32').transpose()
# load data in n x NFFT/2+1 format
# print(np.size(x,0))
# print(np.size(x,1))
# y = make_window_buffer(x)
# print(y)
# print(np.size(y,0))
# print(np.size(y,1))
# z = Normalize_data(y, mu, std)
# print(z)
# print(np.size(z,0))
# print(np.size(z,1))
# z = gen_context(x, 3, mu, std)
# print(z[1,:])
# print(z[-1,:])








# x = np.random.rand(3, 2)
# x[0, :] = [1, 2]
# x[1, :] = [3, 4]
# x[2, :] = [5, 6]
# #y = make_window_buffer(x)
# #print(x)
# #print(y)
#
# mu = np.zeros([2])
# mu[0] = 3
# mu[1] = 2
# std = np.ones([2])
# std[0] = 10
# std[1] = 100
# z = Normalize_data(np.tile(x,[1,2]), mu, std)
# print(z)

