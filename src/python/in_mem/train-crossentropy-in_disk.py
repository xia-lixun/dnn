import numpy as np
import tensorflow as tf
import scipy.io as scio
import h5py
import time
import os, os.path
import psutil
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split





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






TRAIN_ROOT = '/home/coc/workspace/train/'
TEST_ROOT = '/home/coc/workspace/test/'
MODEL_LOCATION = '/home/coc/workspace/model-20180209.mat'


# tensor size and system memory utilization
MEM_UTIL_RATIO = 0.7
MEM_AVAILABLE = psutil.virtual_memory().available
TEST_FILES, TEST_BYTES = dataset_size(TEST_ROOT)
TRAIN_FILES, TRAIN_BYTES = dataset_size(TRAIN_ROOT)
TEST_PARTITIONS = int(TEST_BYTES // (MEM_AVAILABLE * MEM_UTIL_RATIO)) + 1
TRAIN_PARTITIONS = int(TRAIN_BYTES // (MEM_AVAILABLE * MEM_UTIL_RATIO * 0.5)) + 1
print('train: %f MiB in %d partitions'%(TRAIN_BYTES/1024/1024, TRAIN_PARTITIONS))
print('test: %f MiB in %d partitions'%(TEST_BYTES/1024/1024, TEST_PARTITIONS))

# GPU memory utillization
GPU_MEM_AVAILABLE = 11*1024*1024*1024
GPU_MEM_UTIL_RATIO = 0.4
TEST_BATCH_PARTITIONS = int((TEST_BYTES/TEST_PARTITIONS) // (GPU_MEM_AVAILABLE * GPU_MEM_UTIL_RATIO)) + 1
print('test: %d GPU partitions for each CPU partition'%(TEST_BATCH_PARTITIONS))

# find out the tensor dimensions
tensor = loadmat_transpose(os.path.join(TRAIN_ROOT, 't_1.mat'))
INPUT_DIM = tensor['variable'].shape[1]
OUTPUT_DIM = tensor['label'].shape[1]
del tensor
print('input,output dims = %d,%d' % (INPUT_DIM, OUTPUT_DIM))



HIDDEN_LAYER_WIDTH = 2048
N_EPOCHS = 400
BATCH_SIZE_INIT = 128
LEARN_RATE_INIT = 0.01
DROPOUT_COEFF = 0.8
L2_LOSS_COEFF = 0.00
MOMENTUM_COEFF = 0.9
rng = np.random.RandomState(4913)






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



keep_prob = tf.placeholder(tf.float32)
lrate_p = tf.placeholder(tf.float32)
mt_p = tf.placeholder(tf.float32)



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


x = tf.placeholder(tf.float32, [None, INPUT_DIM])
y = f_props(layers, x)
t = tf.placeholder(tf.float32, [None, OUTPUT_DIM])



# cost = tf.reduce_mean(tf.reduce_sum((y - t)**2, 1))
cost_op = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=y))+
          L2_LOSS_COEFF * tf.nn.l2_loss(layers[0].W)+
          L2_LOSS_COEFF * tf.nn.l2_loss(layers[1].W)+
          L2_LOSS_COEFF * tf.nn.l2_loss(layers[2].W)+
          L2_LOSS_COEFF * tf.nn.l2_loss(layers[3].W))
train_op = tf.train.MomentumOptimizer(learning_rate=lrate_p, momentum=mt_p).minimize(cost_op)




# saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())







##########################
##      PROCESSING      ##
##########################

def dataset_dimension(path, select):
    #note: select can be in any order! [7, 3, 9, 11, 4, ...]

    spect_examples = 0
    label_examples = 0
    spect_width = 0
    label_width = 0

    for p in select:

        # temp = scio.loadmat(os.path.join(path, 't_' + str(p) + '.mat'))
        temp = loadmat_transpose(os.path.join(path, 't_' + str(p) + '.mat'))

        spect_examples += temp['variable'].shape[0]
        label_examples += temp['label'].shape[0]
        spect_width = temp['variable'].shape[1]
        label_width = temp['label'].shape[1]
        del temp

    return spect_examples, spect_width, label_examples, label_width




def dataset_load2mem(path, select):
    #note: select can be in any order! [7, 3, 9, 11, 4, ...]

    m, n, p, q = dataset_dimension(path, select)
    spect = np.zeros((m,n), dtype='float32')
    label = np.zeros((p,q), dtype='float32')
    offset = 0

    for i in select:

        #temp = scio.loadmat(os.path.join(path, 't_' + str(i) + '.mat'))
        temp = loadmat_transpose(os.path.join(path, 't_' + str(i) + '.mat'))

        instance = temp['variable'].shape[0]
        spect[offset:offset+instance] = temp['variable']
        label[offset:offset+instance] = temp['label']
        offset += instance
        del temp

    return spect, label




def evaluate_batch_cost(spect, label):

    cost_batch = np.zeros((TEST_BATCH_PARTITIONS))
    for (k,i) in enumerate(np.array_split(range(label.shape[0]), TEST_BATCH_PARTITIONS)):
        cost_batch[k] = sess.run(cost_op, feed_dict={x:spect[i[0]:i[-1]], t:label[i[0]:i[-1]], keep_prob:1.0})
    return np.mean(cost_batch)


def evaluate_total_cost():
    
    total_cost = 0.0

    for portion in np.array_split(range(1,1+len([x for x in os.listdir(TEST_ROOT)])), TEST_PARTITIONS):
        test_spect, test_label = dataset_load2mem(TEST_ROOT, portion)
        # print('spectrum size %d %d' % (test_spect.shape[0], test_spect.shape[1]))
        # print('label size %d %d' %(test_label.shape[0], test_label.shape[1]))
        total_cost += evaluate_batch_cost(test_spect, test_label)
        del test_spect
        del test_label

    return total_cost/TEST_PARTITIONS



def training():

    cost_optimal = 1000000.0
    mt = MOMENTUM_COEFF
    bs_rt = BATCH_SIZE_INIT
    lr_rt = LEARN_RATE_INIT


    cost_validation = evaluate_total_cost()
    print('[init]: validation cost: %.3f ' % (cost_validation))


    for epoch in range(N_EPOCHS):

        print('----------------------------------------------------------------------')        
        #if epoch >= 30:
        #    lr_rt = 0.001
        #if epoch >= 40:
        #    lr_rt = 0.0001

        time_start = time.time()

        for rand_portion in np.array_split(shuffle(range(1,1+len([x for x in os.listdir(TRAIN_ROOT)]))), TEST_PARTITIONS):

            train_spect, train_label = dataset_load2mem(TRAIN_ROOT, rand_portion)
            train_spect, train_label = shuffle(train_spect, train_label)

            for i in range(train_label.shape[0] // bs_rt):
                a = i * bs_rt
                b = a + bs_rt
                sess.run(train_op, feed_dict={x:train_spect[a:b], t:train_label[a:b], keep_prob:DROPOUT_COEFF, lrate_p:lr_rt, mt_p:mt})

            del train_spect
            del train_label

        cost_validation = evaluate_total_cost()
        
        time_end = time.time()
        print('[epoch %i] validation cost = %.3f'%(epoch+1, cost_validation))
        print('[epoch %i] time = %.3f (sec)'%(epoch+1, time_end-time_start))

        if (cost_validation < cost_optimal):
            save_dict = {}
            save_dict['W1'] = sess.run(layers[0].W)
            save_dict['b1'] = sess.run(layers[0].b)
            save_dict['W2'] = sess.run(layers[1].W)
            save_dict['b2'] = sess.run(layers[1].b)
            save_dict['W3'] = sess.run(layers[2].W)
            save_dict['b3'] = sess.run(layers[2].b)
            save_dict['W4'] = sess.run(layers[3].W)
            save_dict['b4'] = sess.run(layers[3].b)

            scio.savemat(MODEL_LOCATION, save_dict)
            cost_optimal = cost_validation
            print('[epoch %d] |||||||||||||||||||| model saved ||||||||||||||||||||' % (epoch + 1))


training()
sess.close()


    # exponential decay (simulated annealing) may converge to 'sharp' global minimum
    # which generalizes poorly. we use hybrid discrete noise scale falling here.
    # "Don't decay the learning rate, increase the batch size, Samuel L. Smith et al. Google Brain"












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




