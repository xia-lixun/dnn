import tensorflow as tf


# graph
dummy_input = tf.random_uniform([5], minval=0, maxval=10, dtype=tf.int32)
dummy_input = tf.Print(dummy_input, data=[dummy_input], message='New dummy inputs created: ', summarize=6)

q = tf.FIFOQueue(capacity=3, dtypes=tf.int32)
enqueue_op = q.enqueue_many(dummy_input)

# setup a queue runner to handle enqueue_op outside of the main thread asynchronously
qr = tf.train.QueueRunner(q, [enqueue_op]*1)
tf.train.add_queue_runner(qr)

data = q.dequeue()
data = tf.Print(data, data=[q.size(), data], message='This is how many items left in q: ')
fg = data + 1


with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    # now dequeue a few times 
    sess.run(fg)
    sess.run(fg)
    sess.run(fg)
    # comsumes all, blocking for data-waiting
    sess.run(fg)
    sess.run(fg)
    sess.run(fg)

    print('We will never be here')
    coord.request_stop()
    coord.join(threads)