import tensorflow as tf
import numpy as np



BM_BINS = 257
SPECTRAL_SPREAD = 24
NUM_EPOCHS = 3
BATCH_SIZE = 100
NUM_THREADS = 4
MIN_AFTER_DEQUEUE = 10000
NUM_VALID_PARTS = 13
NUM_TRAIN_PARTS = 13
NUM_VALID_EXAMPLES = 122719
NUM_TRAIN_EXAMPLES = 122380


rng = np.random.RandomState(42)





def read_data(file_q):

    label_len = BM_BINS
    image_len = BM_BINS * SPECTRAL_SPREAD

    label_bytes = label_len * 4
    image_bytes = image_len * 4

    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(file_q)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_floats = tf.decode_raw(value, tf.float32)

    # The first bytes represent the label, which we convert from uint8->int32.
    image = tf.strided_slice(record_floats, [0], [image_len])
    label = tf.strided_slice(record_floats, [image_len], [image_len + label_len])
    return image, label



def train_run(valid_image, valid_label, train_image, train_label):

    valid_label_hat = f_props(layers, valid_image)
    valid_cost = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=valid_label, logits=valid_label_hat), 1))

    train_label_hat = f_props(layers, train_image)
    train_cost = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_label, logits=train_label_hat), 1))
    train_opt = tf.train.MomentumOptimizer(learning_rate=lrate_p, momentum=mt_p).minimize(train_cost)


    
    with tf.Session() as sess:
            
        sess.run(tf.global_variables_initializer())
        # print(save_dict['B3'])
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        valid_blocks = NUM_VALID_EXAMPLES // BATCH_SIZE
        train_blocks = NUM_TRAIN_EXAMPLES // BATCH_SIZE


        for epoch in range(NUM_EPOCHS):
            mt = 0.9
            lrate = 0.001
            if epoch > 10:
                lrate = 0.0005
            if epoch > 20:
                lrate = 0.00025
            if epoch > 30:
                lrate = 0.000125
            if epoch > 40:
                lrate = 0.0000625

            valid_cost_val = 0.0
            for i in range(valid_blocks):
                valid_cost_val = valid_cost_val + sess.run(valid_cost, feed_dict={keep_prob: 1.0})
                # image_batch, label_batch = sess.run([image, label])
                # print(image_batch.shape, label_batch.shape)
            print(valid_cost_val/valid_blocks)

            for i in range(train_blocks):
                sess.run(train_opt, feed_dict={keep_prob: 0.8, lrate_p: lrate, mt_p: mt})

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
        Dense(BM_BINS * SPECTRAL_SPREAD, 512, tf.nn.sigmoid),
        Dense(512, 512, tf.nn.sigmoid),
        Dense(512, 512, tf.nn.sigmoid),
        Dense(512, BM_BINS)
    ]
keep_prob = tf.placeholder(tf.float32)
lrate_p = tf.placeholder(tf.float32)
mt_p = tf.placeholder(tf.float32)


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
    
    valid_file_list = [valid_path + 'tensor-{}.bin'.format(i) for i in range(NUM_VALID_PARTS)]
    train_file_list = [train_path + 'tensor-{}.bin'.format(i) for i in range(NUM_TRAIN_PARTS)]

    valid_file_q = tf.train.string_input_producer(valid_file_list)
    train_file_q = tf.train.string_input_producer(train_file_list)
    
    valid_image, valid_label = read_data(valid_file_q)
    train_image, train_label = read_data(train_file_q)
    # label = tf.Print(label, data=[file_q.size()], message='Files left in q: ')
    valid_image_b, valid_label_b = tf.train.shuffle_batch([valid_image, valid_label], bsz, cap, mad, num_threads=nths, shapes=shps)
    train_image_b, train_label_b = tf.train.shuffle_batch([train_image, train_label], bsz, cap, mad, num_threads=nths, shapes=shps)

    train_run(valid_image_b, valid_label_b, train_image_b, train_label_b)







shuffle_batch('D:/6-Workspace/Mix/training/tensor/', 'D:/6-Workspace/Mix/test/tensor/')

# x = tf.placeholder(tf.float32, [None, 257*24])
# t = tf.placeholder(tf.float32, [None, 257])

# saver = tf.train.Saver()


