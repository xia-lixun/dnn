import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import scipy.io as scio
import h5py
import time



BM_BINS = 257
BM_SPREAD = 24
HIDDEN_LAYER_WIDTH = 2048

N_EPOCHS = 50
BATCH_SIZE = 5000

TRAIN_ROOT = '/home/coc/5-Workspace/train/tensor_'
TRAIN_PARTS = 38
TEST_ROOT = '/home/coc/5-Workspace/test/tensor_'
TEST_PARTS = 8

MOMENTUM_COEFF = 0.9
LEARN_RATE_INIT = 0.05
MATFILE = '/home/coc/5-Workspace/train/specification-2017-12-01.mat'

rng = np.random.RandomState(1234)
evaluate_cost_opt = 1000000.0




##########################
##      NOT IN USE      ##
##########################
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






##########################
##         GRAPH        ##
##########################
class Dense:

    def __init__(self, in_dim, out_dim, function=lambda x: x):
        self.W = tf.Variable(rng.uniform(low = -0.1, high = 0.1, size=(in_dim, out_dim)).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
        self.function = function
        self.params = [self.W, self.b]
        # self.ae = Autoencoder(in_dim, out_dim, self.W, self.function)

    def f_prop(self, x):
        u = tf.matmul(x, self.W) + self.b
        self.z = self.function(u)
        return self.z

    # def pretrain(self, x, noise):
    #    cost, reconst_x = self.ae.reconst_error(x, noise)
    #    return cost, reconst_x



def f_props(layers, x):
    for i, layer in enumerate(layers):
        x = layer.f_prop(x)
        if(i != len(layers)-1):
            x = tf.nn.dropout(x, keep_prob)
    return x




layers = [
    Dense(BM_BINS*BM_SPREAD, HIDDEN_LAYER_WIDTH, tf.nn.sigmoid),
    Dense(HIDDEN_LAYER_WIDTH, HIDDEN_LAYER_WIDTH, tf.nn.sigmoid),
    Dense(HIDDEN_LAYER_WIDTH, HIDDEN_LAYER_WIDTH, tf.nn.sigmoid),
    Dense(HIDDEN_LAYER_WIDTH, BM_BINS, tf.nn.sigmoid)
]

keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, BM_BINS*BM_SPREAD])
t = tf.placeholder(tf.float32, [None, BM_BINS])
y = f_props(layers, x)
lrate_p = tf.placeholder(tf.float32)
mt_p = tf.placeholder(tf.float32)

# cost = tf.reduce_mean(tf.reduce_sum((y - t)**2, 1))
cost_op = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=y), 1))
train_op = tf.train.MomentumOptimizer(learning_rate=lrate_p, momentum=mt_p).minimize(cost_op)

# saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())


sess.close()





##########################
##      PROCESSING      ##
##########################

def load_dataset_to_mem(path, n_parts):
    # h5 data are in (BM_BINS * BM_SPREAD)-by-Frames shape
    # loaded with h5py with auto tranpose into tensor shape
    data_pool = []
    label_pool = [] 

    for p_ in range(n_parts):
        fid = h5py.File(path + str(p_) + '.h5')
        data = np.array(fid["/data"], dtype='float32')
        label = np.array(fid["/label"], dtype='float32')
        data_pool.append(data)
        label_pool.append(label)
        del label
        del data
        del fid
    data_pool = np.asarray(data_pool)
    label_pool = np.asarray(label_pool)
    return data_pool, label_pool


def evaluate_cost(data_pool, label_pool):
    n = label_pool.shape[0]
    cost_part = np.zeros((n))
    for i in range(n):
        cost_part[i] = sess.run(cost_op, feed_dict={x:data_pool[i], t:label_pool[i], keep_prob:1.0})
    cost_eval = np.mean(cost_part)
    return cost_eval






def training():

    mt = MOMENTUM_COEFF
    lrate = LEARN_RATE_INIT

    test_datapool, test_labelpool = load_dataset_to_mem(TEST_ROOT, TEST_PARTS)
    evaluate_cost_val = evaluate_cost(test_datapool, test_labelpool)
    print('[init]: validation cost: %.3f ' % (evaluate_cost_val))

    for epoch in range(N_EPOCHS):
        
        # simulated annealing
        if(epoch>10):
            lrate = lrate * 0.5
        if(epoch>20):
            lrate = lrate * 0.5
        if(epoch>30):
            lrate = lrate * 0.5
        if(epoch>40):
            lrate = lrate * 0.5
        
        time_start = time.time()
        part_list = shuffle(range(TRAIN_PARTS))
        part_n = part_list.shape[0]
        part_i = 0

        for part_ in part_list:
            fid_train = h5py.File(TRAIN_ROOT + str(part_) + '.h5')
            train_data = np.array(fid_train["/data"], dtype='float32')
            train_label = np.array(fid_train["/label"], dtype='float32')
            del fid_train

            train_data, train_label = shuffle(train_data, train_label)
            n_batch = train_label.shape[0] // BATCH_SIZE

            for i in range(n_batch):
                start = i * BATCH_SIZE
                end = start + BATCH_SIZE
                sess.run(train_op, feed_dict={x:train_data[start:end], t:train_label[start:end], keep_prob:0.8, lrate_p:lrate, mt_p:mt})

            del train_label
            del train_data            
            part_i += 1
            print('part %d/%d finished'%(part_i,part_n))

        evaluate_cost_val = evaluate_cost(test_datapool, test_labelpool)
        time_end = time.time()
        print('[epoch %i] validation cost = %.3f ' % (epoch + 1, evaluate_cost_val))
        print('[epoch %i] time = %.3f (sec)' % (epoch + 1, time_end - time_start))

        if (evaluate_cost_val < evaluate_cost_opt):
            save_dict = {}
            save_dict['W1'] = sess.run(layers[0].W)
            save_dict['b1'] = sess.run(layers[0].b)
            save_dict['W2'] = sess.run(layers[1].W)
            save_dict['b2'] = sess.run(layers[1].b)
            save_dict['W3'] = sess.run(layers[2].W)
            save_dict['b3'] = sess.run(layers[2].b)
            save_dict['W4'] = sess.run(layers[3].W)
            save_dict['b4'] = sess.run(layers[3].b)

            scio.savemat(MATFILE, save_dict)
            evaluate_cost_opt = evaluate_cost_val
            print('[epoch %d] model saved' % (epoch + 1))

    del test_labelpool
    del test_datapool














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

