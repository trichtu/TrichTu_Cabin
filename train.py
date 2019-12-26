from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
from utils import *
from models import GCN
import matplotlib.pyplot as plt
import scipy.sparse as sp
plt.switch_backend('agg')
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"



def train_step(model_file):
    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)
    dirname = FLAGS.dataset+model_file+'/'
    print(dirname)
    # Load data
    namelist_path = FLAGS.dataset+'sub_list_new.txt'
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data( FLAGS.hemisphere, namelist=namelist_path, MPM=True)
    # neighbor_mask = 50*np.load( FLAGS.modeldir+'neighbor_mask_{}.npy'.format(FLAGS.hemisphere))
    namelist= [str(name).replace("\n", "") for name in open(namelist_path, 'r').readlines()]
    # mpm = np.max(np.load( FLAGS.modeldir+'MPM_{}.npy'.format(FLAGS.hemisphere))[:,:210], axis=1)
    # train_mask = np.array([False]*len(mpm))
    # train_mask[mpm>0.5] = True
    # Some preprocessing
    adj = sparse_to_tuple(adj)

    for i in range(len(features)):
        features[i] = preprocess_features(features[i])

    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree



    # Define placeholders/
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports) ],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant( features[0].shape, dtype=tf.float64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(FLAGS.dropout, shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
        'gradient_adj': [tf.sparse_placeholder(tf.float32)],
        'gradient': tf.placeholder(tf.bool)
    }

    # Create model
    model_name='model_{}'.format(FLAGS.hemisphere)
    print('model name:',model_name)
    model = GCN(placeholders, input_dim=features[0].shape[1], layer_num=FLAGS.layer_num, logging=True, name=model_name)

    # Initialize session
    sess = tf.Session()

    # Init variables
    sess.run(tf.global_variables_initializer())

    
    # Define model evaluation function
    def evaluate(features, support, labels, mask, adj, gradient, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, adj, gradient, placeholders)
        outs_val = sess.run([model.loss, model.dice, model.outputs], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)

    def dynamic_learning_rate(epoch):
        # learning_rate = FLAGS.basic_learning_rate * 10**(-3 * epoch/FLAGS.epochs)
        learning_rate = FLAGS.basic_learning_rate * (1 - epoch/FLAGS.epochs)
        return learning_rate

    # def dynamic_training_mask(mpm, epoch):
    #     if (epoch/FLAGS.epochs)>0.5:
    #         train_mask = [True]*len(mpm)
    #     else:
    #         thr = 65-130*epoch/FLAGS.epochs
    #         train_mask = np.array([False]*len(mpm))
    #         train_mask[mpm>thr] = True
    #     return train_mask


    cost_train, acc_train, dc_train = [], [], []
    cost_val, acc_val, dc_val = [], [], []
    # Train model
    print('length',len(features))
    for epoch in range(FLAGS.epochs):
        numlist = np.arange(0, FLAGS.train_num, 1)
        np.random.shuffle(numlist)
        FLAGS.learning_rate = dynamic_learning_rate(epoch)
        # train_mask = dynamic_training_mask(mpm, epoch)
        print('_____leaning rate',FLAGS.learning_rate,'epoch:',epoch)
        for i in numlist:
            t = time.time()
            print('training sample: ', i)
            # Construct feed dictionary
            feed_dict = construct_feed_dict(features[i], support, y_train, train_mask, adj, False, placeholders )

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.dice], feed_dict=feed_dict)
            cost_train.append(outs[1])
            dice_train.append(outs[2])

            # Validation
            validnum = np.random.randint(FLAGS.validate_num)+FLAGS.train_num
            cost_val, dice_val, _, tt = evaluate(features[validnum], support, y_val, val_mask,adj, False, placeholders )
            cost_val.append(cost_val)
            dc_val.append(dice_val)
            # Print results
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                "train_dice=", "{:.3f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost_val),
                "val_dice=", "{:.3f}".format(dice_val), "time=", "{:.5f}".format(tt))

            # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            #     print("Early stopping...")
            #     break
        model.save(sess=sess, path=dirname)

    np.savetxt(dirname+'cost_val.txt', cost_val, fmt='%9.5f', delimiter=',')
    np.savetxt(dirname+'cost_train.txt', cost_train, fmt='%9.5f', delimiter=',')
    np.savetxt(dirname+'dc_val.txt', dc_val, fmt='%9.5f', delimiter=',')
    np.savetxt(dirname+'dc_train.txt', dc_train, fmt='%9.5f', delimiter=',')
    print("Optimization Finished!")

    # model.load(sess)
    # testave = []
    # # Testing
    # for i in range(len(features)):
    #     test_cost, test_acc, test_dice, prediction, test_duration = evaluate(features[i], support, y_test, test_mask, neighbor_mask, placeholders)
    #     print("Test set results:", "cost=", "{:.5f}".format(test_cost),
    #         "accuracy=", "{:.3f}_{:.3f}".format(test_acc, test_dice), "time=", "{:.5f}".format(test_duration))
    #     if i>=FLAGS.train_num:
    #         testave.append(test_dice)
    #     # print(prediction.shape)
    #     path = dirname+'{}.npy'.format(namelist[i])
    #     np.save(path, prediction)
    # np.savetxt(dirname+'dc_test.txt', testave, fmt='%7.4f', delimiter=',')
    # print('average_test_dice:', np.array(testave).mean())

    # Testing and resting
    # #load data
    # f = open('/DATA/232/lma/data/HCP_test/sub_list.txt','r')
    # namelist = [str(int(name)) for name in f.readlines()]
    # feature_path_list = []
    # for name in namelist:
    #     feature_path_list.append('/DATA/232/lma/data/HCP_test/{}/{}_{}_probtrackx_omatrix2/finger_print_fiber.npz'.format(name, name, FLAGS.hemisphere))
    
    # adj, features2, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.hemisphere,pathlist= feature_path_list)
    # # Some preprocessing
    # for i in range(len(features2)):
    #     features2[i] = preprocess_features(features2[i])
    # # prediction
    # testave = []
    # for i in range(len(features2)):
    #     test_cost, test_acc, test_dice, prediction, test_duration = evaluate(features2[i], support, y_test, test_mask, neighbor_mask,placeholders)
    #     print("Test set results:", "cost=", "{:.5f}".format(test_cost),
    #         "accuracy=", "{:.5f}".format(test_dice), "time=", "{:.5f}".format(test_duration))
    #     testave.append(test_dice)
    #     # print(prediction.shape)
    #     path = dirname+'test_{}.npy'.format(namelist[i])
    #     np.save(path, prediction)
    # print('average_test/retest_acc:', np.array(testave).mean())    

    # #Resesting
    # f = open('/DATA/232/lma/data/HCP_retest/sub_list.txt','r')
    # namelist = [str(int(name)) for name in f.readlines()]
    # feature_path_list = []
    # for name in namelist:
    #     feature_path_list.append('/DATA/232/lma/data/HCP_retest/{}/{}_{}_probtrackx_omatrix2/finger_print_fiber.npz'.format(name, name, FLAGS.hemisphere))
    
    # adj, features3, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.hemisphere, pathlist= feature_path_list)
    # # Some preprocessing
    # for i in range(len(features3)):
    #     features3[i] = preprocess_features(features3[i])
    # #prediction
    # testave = []
    # for i in range(len(features3)):
    #     test_cost, test_acc, test_dice, prediction, test_duration = evaluate(features3[i], support, y_test, test_mask, neighbor_mask, placeholders)
    #     print("Test set results:", "cost=", "{:.5f}".format(test_cost),
    #         "accuracy=", "{:.5f}".format(test_dice), "time=", "{:.5f}".format(test_duration))
    #     testave.append(test_dice)
    #     # print(prediction.shape)
    #     path = dirname+'retest_{}.npy'.format(namelist[i])
    #     np.save(path, prediction)
    # print('average_test/retest_acc:', np.array(testave).mean())  
    return None

if __name__ == '__main__':
    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('dataset', '/DATA/232/lma/data/HCPwork/', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
    flags.DEFINE_string('modeldir', '/DATA/232/lma/script/individual-master/GCN-master/', 'Dataset string.')
    flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_float('basic_learning_rate', 0.01, 'basic learning rate.')
    flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
    flags.DEFINE_integer('train_num', 100, 'sample size for training ')
    flags.DEFINE_integer('validate_num', 20, 'sample size for training ')
    flags.DEFINE_integer('layer_num', 1, 'sample size for training ')
    flags.DEFINE_string('hemisphere', 'L', 'cerebral cortex part')
    
    t_test = time.time()
    print('hemisphere: ', FLAGS.hemisphere)
    print('_____layer_num:', FLAGS.layer_num, 'model:', FLAGS.model, 'max_degree:', FLAGS.max_degree, 'hidden:', FLAGS.hidden1)
    model_file='tf_grad_{}'.format(FLAGS.hemisphere)
    train_step(model_file)
    print('_____layer_num:', FLAGS.layer_num, 'model:', FLAGS.model, 'max_degree:', FLAGS.max_degree, 'hidden:', FLAGS.hidden1)
    print('hemisphere: ',FLAGS.hemisphere)
    print('time = ', time.time() - t_test)
