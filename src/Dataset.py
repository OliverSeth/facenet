import numpy as np
import tensorflow as tf
import gzip
import os
import platform
import pickle
import facenet
from tensorflow.python.ops import data_flow_ops


class DataSet(object):
    def __init__(self,index, dtype=tf.float32):
        dype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype {}, expected uint8 or float32'.format(dtype))

        self.index=index
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None
        self.train_data_size = None
        self.test_data_size = None
        self.learning_rate_placeholder=None
        self.batch_size_placeholder=None
        self.phase_train_placeholder=None
        self.image_paths_placeholder=None
        self.labels_placeholder=None
        self.enqueue_op=None
        self.input_queue=None

        self._index_in_train_epoch = 0

        self.facenet_dataset_constructor(index)


    def facenet_dataset_constructor(self,index):
        data_dir='/home/facenet/data/lfw-dataset/lfw-set'+str(index)
        train_set = facenet.get_dataset(data_dir)
        self.learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        self.batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

        self.phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        self.image_paths_placeholder = tf.placeholder(tf.string, shape=(None,3), name='image_paths')
        self.labels_placeholder = tf.placeholder(tf.int64, shape=(None,3), name='labels')

        self.input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string, tf.int64],
                                              shapes=[(3,), (3,)],
                                              shared_name=None, name=None)
        self.enqueue_op = self.input_queue.enqueue_many([self.image_paths_placeholder, self.labels_placeholder])

        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = self.input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)

                #pylint: disable=no-member
                image.set_shape((160, 160, 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])

        image_batch, labels_batch = tf.train.batch_join(
            images_and_labels, batch_size=self.batch_size_placeholder,
            shapes=[(160, 160, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * 30,
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        labels_batch = tf.identity(labels_batch, 'label_batch')
        self.train_data=image_batch
        self.train_label=labels_batch


    def next_batch(self, batch_size):
        start = self._index_in_train_epoch
        self._index_in_train_epoch += batch_size
        if self._index_in_train_epoch > self.train_data_size:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = self.train_data[order]
            self.train_label = self.train_label[order]

            start = 0
            self._index_in_train_epoch = batch_size
            assert batch_size <= self.train_data_size
        end = self._index_in_train_epoch
        return self.train_data[start: end], self.train_label[start: end]


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
