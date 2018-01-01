import tensorflow as tf
import numpy as np




def read_data(file_q):
    label_len = 2
    image_len = 4

    label_bytes = label_len * 4
    image_bytes = image_len * 4

    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(file_q)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_floats = tf.decode_raw(value, tf.float32)

    # The first bytes represent the label, which we convert from uint8->int32.
    label = tf.strided_slice(record_floats, [0], [label_len])
    image = tf.strided_slice(record_floats, [label_len], [label_len + image_len])

    return image, label


def cifar_run(image, label):
    for rounds in range(2):
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for i in range(6):
                image_batch, label_batch = sess.run([image, label])
                # print(image_batch.shape, label_batch.shape)
                print(i, image_batch, label_batch)
            coord.request_stop()
            coord.join(threads)








def cifar_shuffle_batch(data_path):
    batch_size = 4
    num_threads = 4
    # create a list of all our filenames
    filename_list = [data_path + 'data_batch_{}.bin'.format(i + 1) for i in range(5)]
    # create a filename queue
    file_q = tf.train.string_input_producer(filename_list, num_epochs=1)
    # read the data - this contains a FixedLengthRecordReader object which handles the
    # de-queueing of the files.  It returns a processed image and label, with shapes
    # ready for a convolutional neural network
    image, label = read_data(file_q)
    label = tf.Print(label, data=[file_q.size()], message='Files left in q: ')
    shapes = [[4], [2]]
    # setup minimum number of examples that can remain in the queue after dequeuing before blocking
    # occurs (i.e. enqueuing is forced) - the higher the number the better the mixing but
    # longer initial load time
    min_after_dequeue = 10
    # setup the capacity of the queue - this is based on recommendations by TensorFlow to ensure good mixing
    capacity = min_after_dequeue + (num_threads + 1) * batch_size
    # image_batch, label_batch = cifar_shuffle_queue_batch(image, label, batch_size, num_threads)
    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size, capacity, min_after_dequeue, num_threads=num_threads, shapes=shapes)
    # now run the training
    cifar_run(image_batch, label_batch)





cifar_shuffle_batch('D:/')