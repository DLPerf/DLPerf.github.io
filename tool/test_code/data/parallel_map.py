from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import time

import tensorflow as tf


def disk_image_batch_dataset(img_paths, batch_size, shuffle=True, buffer_batch=128, repeat=-1):

    def parse_func(path):
        img = tf.read_file(path)
        img = tf.image.decode_png(img)
        return img

    dataset = tf.data.Dataset.from_tensor_slices(img_paths).map(parse_func)

    dataset = dataset.batch(batch_size)

    if shuffle:
        dataset = dataset.shuffle(buffer_batch)
    else:
        dataset = dataset.prefetch(buffer_batch)

    dataset = dataset.repeat(repeat)

    # make iterator
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


def disk_image_batch_queue(img_paths, batch_size, shuffle=True, num_threads=4, min_after_dequeue=100):

    _, img = tf.WholeFileReader().read(
        tf.train.string_input_producer(img_paths, shuffle=shuffle, capacity=len(img_paths)))
    img = tf.image.decode_png(img)
    img.set_shape([218, 178, 3])

    # batch datas
    if shuffle:
        capacity = min_after_dequeue + (num_threads + 1) * batch_size
        img_batch = tf.train.shuffle_batch([img],
                                           batch_size=batch_size,
                                           capacity=capacity,
                                           min_after_dequeue=min_after_dequeue,
                                           num_threads=num_threads)
    else:
        img_batch = tf.train.batch([img], batch_size=batch_size)

    return img_batch


paths = glob.glob('img_align_celeba/*.jpg')

with tf.Session() as sess:
    with tf.device('/cpu:0'):
        batch = disk_image_batch_dataset(paths, 128, shuffle=True, buffer_batch=128, repeat=-1)
        for _ in range(10):
            start = time.time()
            for _ in range(100):
                sess.run(batch)
            elapse = time.time() - start
            print('Dataset Average: %f ms' % (elapse / 100.0 * 1000))

        batch = disk_image_batch_queue(paths, 128, shuffle=True, num_threads=4, min_after_dequeue=100)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for _ in range(10):
            start = time.time()
            for _ in range(100):
                sess.run(batch)
            elapse = time.time() - start
            print('Queue Average: %f ms' % (elapse / 100.0 * 1000))
        coord.request_stop()
        coord.join(threads)