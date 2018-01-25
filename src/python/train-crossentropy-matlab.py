import numpy as np
import tensorflow as tf
import scipy.io as scio
import h5py
import time
import os, os.path
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split





TRAIN_ROOT = '/media/coc/Dataset/train/'
TEST_ROOT = '/media/coc/Dataset/test/'
MATFILE = '/media/coc/Dataset/model-20180124.mat'
PARTITIONS = 4

BM_BINS = 257
BM_SPREAD = 24
HIDDEN_LAYER_WIDTH = 2048

N_EPOCHS = 100
BATCH_SIZE_INIT = 1000
LEARN_RATE_INIT = 0.01
DROPOUT_COEFF = 0.8
L2_LOSS_COEFF = 0.00
MOMENTUM_COEFF = 0.9




TRAIN_PARTS = len([name for name in os.listdir(TRAIN_ROOT)])
TEST_PARTS = len([name for name in os.listdir(TEST_ROOT)])
print(TRAIN_PARTS)
print(TEST_PARTS)
rng = np.random.RandomState(842)





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
    Dense(HIDDEN_LAYER_WIDTH, BM_BINS)
]

keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, BM_BINS*BM_SPREAD])
t = tf.placeholder(tf.float32, [None, BM_BINS])
y = f_props(layers, x)
lrate_p = tf.placeholder(tf.float32)
mt_p = tf.placeholder(tf.float32)

# cost = tf.reduce_mean(tf.reduce_sum((y - t)**2, 1))
cost_op = (tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=y), 1)) + 
          (L2_LOSS_COEFF * tf.nn.l2_loss(layers[0].W)) +
          (L2_LOSS_COEFF * tf.nn.l2_loss(layers[1].W)) +
          (L2_LOSS_COEFF * tf.nn.l2_loss(layers[2].W)) +
          (L2_LOSS_COEFF * tf.nn.l2_loss(layers[3].W)))
train_op = tf.train.MomentumOptimizer(learning_rate=lrate_p, momentum=mt_p).minimize(cost_op)

# saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())







##########################
##      PROCESSING      ##
##########################

def load_dataset_to_mem(path, part_list):

    temp = scio.loadmat(path + 't_' + str(part_list[0]) + '.mat')
    spect = np.array(temp['spec'], dtype='float32')
    label = np.array(temp['bm'], dtype='float32')
    del temp

    for p_ in range(1, part_list.shape[0]):

        temp = scio.loadmat(path + 't_' + str(part_list[p_]) + '.mat')
        temp_spect = np.array(temp['spec'], dtype='float32')
        temp_label = np.array(temp['bm'], dtype='float32')
        
        spect = np.concatenate((spect,temp_spect))
        label = np.concatenate((label,temp_label))

        del temp_label
        del temp_spect
        del temp

    return spect, label




def evaluate_cost(spect, label):

        cost_value = sess.run(cost_op, feed_dict={x:spect, t:label, keep_prob:1.0})
        return cost_value




def training():

    evaluate_cost_opt = 1000000.0
    mt = MOMENTUM_COEFF
    lrate = LEARN_RATE_INIT
    lbs = BATCH_SIZE_INIT

    test_datapool, test_labelpool = load_dataset_to_mem(TEST_ROOT, shuffle(range(1,1+TEST_PARTS)))
    evaluate_cost_val = evaluate_cost(test_datapool, test_labelpool)
    print('[init]: validation cost: %.3f ' % (evaluate_cost_val))

    for epoch in range(N_EPOCHS):
        
        # exponential decay (simulated annealing) may converge to 'sharp' global minimum
        # which generalizes poorly. we use hybrid discrete noise scale falling here.
        if epoch >= 20:
            lbs = 2000
        if epoch >= 40:
            lbs = 4000
        if epoch >= 60:
            lbs = 8000
        if epoch >= 70:
            lrate = 0.001
        if epoch >= 80:
            lrate = 0.0001
        if epoch >= 90:
            lrate = 0.00001
        
        time_start = time.time()
        part_list = shuffle(range(1,1+TRAIN_PARTS))
        part_n = len(part_list)
        part_i = 0

        part_list_breakout = np.array_split(part_list, PARTITIONS)
        for part_ in part_list_breakout:
            
            train_data, train_label = load_dataset_to_mem(TRAIN_ROOT, part_)
            train_data, train_label = shuffle(train_data, train_label)
            n_batch = train_label.shape[0] // lbs

            for i in range(n_batch):
                start = i * lbs
                end = start + lbs
                sess.run(train_op, feed_dict={x:train_data[start:end], t:train_label[start:end], keep_prob:DROPOUT_COEFF, lrate_p:lrate, mt_p:mt})

            del train_label
            del train_data            
            part_i += 1
            print('...%d/%d'%(part_i,part_n))

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




training()
sess.close()








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




