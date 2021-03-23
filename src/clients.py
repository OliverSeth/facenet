import numpy as np
import tensorflow as tf
import facenet
from Dataset import DataSet
import train_tripletloss


class user(object):
    def __init__(self, localData, localLabel, isToPreprocess):
        self.dataset = localData
        self.label = localLabel
        self.train_dataset = None
        self.train_label = None
        self.isToPreprocess = isToPreprocess

        self.dataset_size = localData.shape[0]
        self._index_in_train_epoch = 0
        self.parameters = {}

        self.train_dataset = self.dataset
        self.train_labellabel = self.label
        if self.isToPreprocess == 1:
            self.preprocess()


    def next_batch(self, batchsize):
        start = self._index_in_train_epoch
        self._index_in_train_epoch += batchsize
        if self._index_in_train_epoch > self.dataset_size:
            order = np.arange(self.dataset_size)
            np.random.shuffle(order)
            self.train_dataset = self.dataset[order]
            self.train_label = self.label[order]
            if self.isToPreprocess == 1:
                self.preprocess()
            start = 0
            self._index_in_train_epoch = batchsize
        end = self._index_in_train_epoch
        return self.train_dataset[start:end], self.train_label[start:end]

    def preprocess(self):
        new_images = []
        shape = (24, 24, 3)
        for i in range(self.dataset_size):
            old_image = self.train_dataset[i, :, :, :]
            old_image = np.pad(old_image, [[4, 4], [4, 4], [0, 0]], 'constant')
            left = np.random.randint(old_image.shape[0] - shape[0] + 1)
            top = np.random.randint(old_image.shape[1] - shape[1] + 1)
            new_image = old_image[left: left + shape[0], top: top + shape[1], :]

            if np.random.random() < 0.5:
                new_image = cv2.flip(new_image, 1)

            mean = np.mean(new_image)
            std = np.max([np.std(new_image),
                          1.0 / np.sqrt(self.train_dataset.shape[1] * self.train_dataset.shape[2] * self.train_dataset.shape[3])])
            new_image = (new_image - mean) / std

            new_images.append(new_image)

        self.train_dataset = new_images



class clients(object):
    def __init__(self,numOfClients,  bLocalBatchSize,
                 eLocalEpoch, sess, train,graph):
        self.num_of_clients = numOfClients
        self.dataset_size = None
        self.test_data = None
        self.test_label = None
        self.B = bLocalBatchSize
        self.E = eLocalEpoch
        self.session = sess
        self.train = train
        self.clientsSet = {}
        self.graph=graph
        self.epoch_size=100

        self.getDataset()


    def getDataset(self):
        for i in range(self.num_of_clients):
            dataset = DataSet(i)
            self.clientsSet['client{}'.format(i)]=dataset


    def ClientUpdate(self, client, global_vars,network,args):
        all_vars = tf.trainable_variables()
        for variable, value in zip(all_vars, global_vars):
            variable.load(value, self.session)

        for i in range(self.E):
            for j in range(self.clientsSet[client].dataset_size // self.B):
                prelogits, _ = network.inference(self.clientsSet[client].image_batch, 1,
                                                 phase_train=self.clientsSet[client].phase_train_placeholder, bottleneck_layer_size=128,
                                                 weight_decay=0.0)

                embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
                anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1,3,128]), 3, 1)
                triplet_loss = facenet.triplet_loss(anchor, positive, negative, 0.2)
                learning_rate = tf.train.exponential_decay(self.clientsSet[client].learning_rate_placeholder, self.graph.global_step,
                                                           100*self.epoch_size, 1.0, staircase=True)
                tf.summary.scalar('learning_rate', learning_rate)
                regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')
                train_op = facenet.train(total_loss, self.graph.global_step, 'ADAGRAD',
                                         learning_rate, 0.9999, tf.global_variables())
                self.session.run(tf.global_variables_initializer(), feed_dict={self.clientsSet[client].phase_train_placeholder:True})
                self.session.run(tf.local_variables_initializer(), feed_dict={self.clientsSet[client].phase_train_placeholder:True})
                coord = tf.train.Coordinator()
                tf.train.start_queue_runners(coord=coord, sess=self.session)
                step = self.session.run(self.graph.global_step, feed_dict=None)
                epoch = step // self.B
                train_tripletloss.train(args, self.session, self.clientsSet[client].train_data, epoch, self.clientsSet[client].image_paths_placeholder, self.clientsSet[client].labels_placeholder, self.clientsSet[client].train_label,
                                        self.clientsSet[client].batch_size_placeholder, self.clientsSet[client].learning_rate_placeholder, self.clientsSet[client].phase_train_placeholder, self.clientsSet[client].enqueue_op, input_queue, self.graph.global_step,
                      embeddings, total_loss, train_op, tf.summary.merge_all(), tf.summary.FileWriter('/log', self.graph), args.learning_rate_schedule_file,
                      128, anchor, positive, negative, triplet_loss)


        return self.session.run(tf.trainable_variables())