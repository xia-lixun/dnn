import tensorflow as tf
import numpy as np
import scipy.io as scio
import time




BM_BINS = 257
SPECTRAL_SPREAD = 24
HIDDEN_WIDTH = 2048

LEARN_INIT_RATE = 0.05
LEARN_DECAY_RATE = 0.97
MOMENTUM_COEFF = 0.9

NUM_EPOCHS = 200
BATCH_SIZE = 1000
NUM_THREADS = 6
MIN_AFTER_DEQUEUE = 50000
NUM_VALID_PARTS = 8
NUM_TRAIN_PARTS = 38
NUM_VALID_EXAMPLES = 1239719
NUM_TRAIN_EXAMPLES = 6187742

rng = np.random.RandomState(42)








def read_data(file_q):

    label_len = BM_BINS
    image_len = BM_BINS * SPECTRAL_SPREAD

    label_bytes = label_len * 4
    image_bytes = image_len * 4
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(file_q)
    record_floats = tf.decode_raw(value, tf.float32)

    image = tf.strided_slice(record_floats, [0], [image_len])
    label = tf.strided_slice(record_floats, [image_len], [image_len + label_len])
    return image, label



def train_run(valid_image, valid_label, train_image, train_label):

    valid_label_hat = f_props(layers, valid_image)
    valid_cost = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=valid_label, logits=valid_label_hat), 1))

    train_label_hat = f_props(layers, train_image)
    train_cost = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_label, logits=train_label_hat), 1))
    train_opt = tf.train.MomentumOptimizer(learning_rate=lrate_p, momentum=mt_p).minimize(train_cost)
    valid_cost_opt = 1000000.0
    

    with tf.Session() as sess:
            
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        valid_blocks = NUM_VALID_EXAMPLES // BATCH_SIZE
        train_blocks = NUM_TRAIN_EXAMPLES // BATCH_SIZE

        mt = MOMENTUM_COEFF
        lrate = LEARN_INIT_RATE/LEARN_DECAY_RATE

        for epoch in range(NUM_EPOCHS):
	    
            time_start = time.time()
            lrate = lrate * LEARN_DECAY_RATE

	    # calculate validation
            valid_cost_val = 0.0
            for i in range(valid_blocks):
                valid_cost_val = valid_cost_val + sess.run(valid_cost, feed_dict={keep_prob: 1.0})
                # image_batch, label_batch = sess.run([image, label])
            valid_cost_val = valid_cost_val/valid_blocks
            print('[epoch %d] validation cost: %.3f ' % (epoch + 1, valid_cost_val))	    

            if (valid_cost_val < valid_cost_opt):
                save_dict = {}
                save_dict['W1'] = sess.run(layers[0].W)
                save_dict['b1'] = sess.run(layers[0].b)
                save_dict['W2'] = sess.run(layers[1].W)
                save_dict['b2'] = sess.run(layers[1].b)
                save_dict['W3'] = sess.run(layers[2].W)
                save_dict['b3'] = sess.run(layers[2].b)
                save_dict['W4'] = sess.run(layers[3].W)
                save_dict['b4'] = sess.run(layers[3].b)
                save_dict['W5'] = sess.run(layers[4].W)
                save_dict['b5'] = sess.run(layers[4].b)
                save_dict['W6'] = sess.run(layers[5].W)
                save_dict['b6'] = sess.run(layers[5].b)

                MATFILE = '/home/coc/5-Workspace/train/model_20180102.mat'
                scio.savemat(MATFILE, save_dict)
                valid_cost_opt = valid_cost_val
                print('[epoch %d] model saved' % (epoch + 1))

            # update W and b
            for i in range(train_blocks):
                sess.run(train_opt, feed_dict={keep_prob: 0.8, lrate_p: lrate, mt_p: mt})
            time_end = time.time()
            print('[epoch %d] took [%.3f] sec' % (epoch + 1, time_end - time_start))

        coord.request_stop()
        coord.join(threads)




class Dense:
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        self.W = tf.Variable(rng.uniform(low = -0.1, high = 0.1, size=(in_dim, out_dim)).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
        self.function = function
        self.params = [self.W, self.b]

    def f_prop(self, x):
        u = tf.matmul(x, self.W) + self.b
        self.z = self.function(u)
        return self.z



layers = [
        Dense(BM_BINS * SPECTRAL_SPREAD, HIDDEN_WIDTH, tf.nn.sigmoid),
        Dense(HIDDEN_WIDTH, HIDDEN_WIDTH, tf.nn.sigmoid),
        Dense(HIDDEN_WIDTH, HIDDEN_WIDTH, tf.nn.sigmoid),
        Dense(HIDDEN_WIDTH, HIDDEN_WIDTH, tf.nn.sigmoid),
        Dense(HIDDEN_WIDTH, HIDDEN_WIDTH, tf.nn.sigmoid),
        Dense(HIDDEN_WIDTH, BM_BINS)
    ]

keep_prob = tf.placeholder(tf.float32)
lrate_p = tf.placeholder(tf.float32)
mt_p = tf.placeholder(tf.float32)
#saver = tf.train.Saver()


def f_props(layers, x):
    for i, layer in enumerate(layers):
        x = layer.f_prop(x)
        if(i != len(layers)-1):
            x = tf.nn.dropout(x, keep_prob)
    return x



def shuffle_batch(train_path, valid_path):

    bsz = BATCH_SIZE
    nths = NUM_THREADS
    shps = [[BM_BINS * SPECTRAL_SPREAD], [BM_BINS]]
    mad = MIN_AFTER_DEQUEUE
    cap = mad + (nths + 1) * bsz
    
    valid_file_list = [valid_path + 'tensor_{}.bin'.format(i) for i in range(NUM_VALID_PARTS)]
    train_file_list = [train_path + 'tensor_{}.bin'.format(i) for i in range(NUM_TRAIN_PARTS)]

    valid_file_q = tf.train.string_input_producer(valid_file_list)
    train_file_q = tf.train.string_input_producer(train_file_list)
    
    valid_image, valid_label = read_data(valid_file_q)
    train_image, train_label = read_data(train_file_q)
    # label = tf.Print(label, data=[file_q.size()], message='Files left in q: ')
    valid_image_b, valid_label_b = tf.train.shuffle_batch([valid_image, valid_label], bsz, cap, mad, num_threads=nths, shapes=shps)
    train_image_b, train_label_b = tf.train.shuffle_batch([train_image, train_label], bsz, cap, mad, num_threads=nths, shapes=shps)

    train_run(valid_image_b, valid_label_b, train_image_b, train_label_b)







shuffle_batch('/home/coc/5-Workspace/train/', '/home/coc/5-Workspace/test/')





