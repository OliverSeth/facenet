from train_tripletloss import *

from __future__ import absolute_import, division, print_function

import collections
from six.moves import range
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from clients import clients, user


def federated_train(args):
    network = importlib.import_module(args.model_def)
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
    with tf.Graph().as_default() as graph:
        tf.set_random_seed(args.seed)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        with sess.as_default():
            sess.run(tf.initialize_all_variables())
            myClients = clients(args.num_of_clients, args.batchsize, args.epoch, sess, train, graph)
            vars = tf.trainable_variables()
            global_vars = sess.run(vars)
            for i in range(args.num_of_clients):
                order = np.arange(args.num_of_clients)
                np.random.shuffle(order)
                clients_in_comm = ['client{}'.format(i) for i in order[0:args.num_of_clients]]
                sum_vars = None
                for client in tqdm(clients_in_comm):
                    local_vars = myClients.ClientUpdate(client, global_vars)
                    if sum_vars is None:
                        sum_vars = local_vars
                    else:
                        for sum_var, local_var in zip(sum_vars, local_vars):
                            sum_var += local_var
                global_vars = []
                for var in sum_vars:
                    global_vars.append(var / args.num_of_clients)
            saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
            save_variables_and_metagraph(sess, saver, tf.summary, model_dir, subdir,
                                         sess.run(tf.Variable(0, trainable=False), feed_dict=None))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.',
                        default='/home/facenet/src/models')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        default='/home/facenet/data/lfw-dataset/lfw_train')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=100)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--people_per_batch', type=int,
                        help='Number of people per batch.', default=45)
    parser.add_argument('--images_per_person', type=int,
                        help='Number of images per person.', default=40)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=500)
    parser.add_argument('--alpha', type=float,
                        help='Positive to negative triplet distance margin.', default=0.2)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop',
                        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
                             'If the size of the images in the data directory is equal to image_size no cropping is performed',
                        action='store_true')
    parser.add_argument('--random_flip',
                        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
                        default='data/learning_rate_schedule.txt')

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.',
                        default='/home/facenet/data/lfw-dataset/lfw_funneled/pairs.txt')
    parser.add_argument('--lfw_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        default='/home/facenet/data/lfw-dataset/lfw_funneled')
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)


if __name__ == '__main__':
    federated_train(parse_arguments(sys.argv[1:]))
