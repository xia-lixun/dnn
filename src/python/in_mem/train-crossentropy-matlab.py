import numpy as np
import tensorflow as tf
import scipy.io as scio
import h5py
import time
import os, os.path
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split





TRAIN_ROOT = '/home/coc/6-Workspace/train/'
TEST_ROOT = '/home/coc/6-Workspace/test/'
MATFILE = '/home/coc/6-Workspace/model-20180131.mat'


# due to 11GB size limit of the GPU mem
# divide the dataset to fit the size
TEST_PARTITIONS = 2

INPUT_DIM = (257+26)*24
OUTPUT_DIM = 257
HIDDEN_LAYER_WIDTH = 2048

N_EPOCHS = 100
BATCH_SIZE_INIT = 1000
LEARN_RATE_INIT = 0.01
DROPOUT_COEFF = 0.8
L2_LOSS_COEFF = 0.00
MOMENTUM_COEFF = 0.9

RAND_SEED = 42
rng = np.random.RandomState(RAND_SEED)



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
    Dense(INPUT_DIM, HIDDEN_LAYER_WIDTH, tf.nn.sigmoid),
    Dense(HIDDEN_LAYER_WIDTH, HIDDEN_LAYER_WIDTH, tf.nn.sigmoid),
    Dense(HIDDEN_LAYER_WIDTH, HIDDEN_LAYER_WIDTH, tf.nn.sigmoid),
    Dense(HIDDEN_LAYER_WIDTH, OUTPUT_DIM)
]

keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, INPUT_DIM])
t = tf.placeholder(tf.float32, [None, OUTPUT_DIM])
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




def evaluate_cost(spect, label):

    r = np.array_split(range(label.shape[0]), TEST_PARTITIONS)
    cost_part = np.zeros((TEST_PARTITIONS))
    for i in range(TEST_PARTITIONS):
        cost_part[i] = sess.run(cost_op, feed_dict={x:spect[r[i][0]:r[i][0]+len(r[i])], t:label[r[i][0]:r[i][0]+len(r[i])], keep_prob:1.0})
    cost_value = np.mean(cost_part)
    return cost_value






def training():

    cost_opt = 1000000.0

    mt = MOMENTUM_COEFF
    lbs = BATCH_SIZE_INIT
    lrate = LEARN_RATE_INIT

    train_spect, train_label = dataset_load2mem(TRAIN_ROOT)
    print('training dataset loaded into memory')
    print('spectrum size %d %d' % (train_spect.shape[0], train_spect.shape[1]))
    print('label size %d %d' %(train_label.shape[0], train_label.shape[1]))

    test_spect, test_label = dataset_load2mem(TEST_ROOT)
    print('testing dataset loaded into memory')
    print('spectrum size %d %d' % (test_spect.shape[0], test_spect.shape[1]))
    print('label size %d %d' %(test_label.shape[0], test_label.shape[1]))

    cost_val = evaluate_cost(test_spect, test_label)
    print('[init]: validation cost: %.3f ' % (cost_val))


    for epoch in range(N_EPOCHS):
        print('----------------------------------------------------------------------')
        # exponential decay (simulated annealing) may converge to 'sharp' global minimum
        # which generalizes poorly. we use hybrid discrete noise scale falling here.
        # "Don't decay the learning rate, increase the batch size, Samuel L. Smith et al. Google Brain"
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
        train_spect, train_label = shuffle(train_spect, train_label)
        n_batch = train_label.shape[0] // lbs

        for i in range(n_batch):
            start = i * lbs
            end = start + lbs
            sess.run(train_op, feed_dict={x:train_spect[start:end], t:train_label[start:end], keep_prob:DROPOUT_COEFF, lrate_p:lrate, mt_p:mt})

        cost_val = evaluate_cost(test_spect, test_label)
        time_end = time.time()
        print('[epoch %i] validation cost = %.3f'%(epoch+1, cost_val))
        print('[epoch %i] time = %.3f (sec)'%(epoch+1, time_end-time_start))

        if (cost_val < cost_opt):
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
            cost_opt = cost_val
            print('[epoch %d] model saved' % (epoch + 1))
        print('----------------------------------------------------------------------')

    del train_spect
    del train_label
    del test_label
    del test_spect




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




