""" Using TF to train a DNN regressor for Boston Housing Data.

ref: http://cs.smith.edu/dftwiki/images/b/bd/TFLinearRegression_BostonData.pdf
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics

args = None

def model(inputs, layers, act):
    [m1, m2, m3] = layers
    #y1 = tf.add(tf.matmul(inputs, m1), b1)
    y1 = tf.matmul(inputs, m1)
    y1 = act(y1)

    y2 = tf.matmul(y1, m2)
    y2 = act(y2)
    
    y3 = tf.matmul(y2, m3)
    #y3 = act(y3)

    #y4 = tf.add(tf.matmul(y3, m4), b4)
    #y4 = act(y4)

    #y_ret = tf.matmul(y4, m_out) + b_out
    return y3

def main():
    boston = learn.datasets.load_dataset('boston')
    x, y = boston.data, boston.target
    y.resize(y.size, 1)
    train_x, test_x, train_y, test_y = train_test_split(
            x, y, test_size = .2, random_state=int(np.random.rand(1)))
    print 'train: {}/{}, test: {}/{}'.format(len(train_x), len(x), len(test_x), len(x))
    print 'dimension of data: {}'.format(x.shape)
    
    # scale the data to (0, 1).
    scaler = preprocessing.StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.fit_transform(test_x)
    num_features = train_x.shape[1]
    print 'num of features: {}'.format(num_features)

    with tf.name_scope('IO'):
        inputs = tf.placeholder(tf.float32, [None, num_features], name='X')
        outputs = tf.placeholder(tf.float32, [None, 1], name='Yhat')
    
    with tf.name_scope('LAYER'):
        # DNN architecture
        #layers = [num_features, 8, 8, 8, 8, 1]
        layers = [num_features, 8, 8, 1]
        # Weight matrices
        m1 = tf.Variable(tf.random_normal([layers[0], layers[1]], 0, .1, dtype=tf.float32), name='m1')
        m2 = tf.Variable(tf.random_normal([layers[1], layers[2]], 0, .1, dtype=tf.float32), name='m2')
        m3 = tf.Variable(tf.random_normal([layers[2], layers[3]], 0, .1, dtype=tf.float32), name='m3')
        #m4 = tf.Variable(tf.random_normal([layers[3], layers[4]], 0, .1, dtype=tf.float32), name='m4')
        #m_out = tf.Variable(tf.random_normal([layers[4], layers[5]],
        #    0, .1, dtype=tf.float32), name='m_out')
        # Bias
        #b1 = tf.Variable(tf.random_normal([layers[1]], 0, .1, dtype=tf.float32), name='b1')
        #b2 = tf.Variable(tf.random_normal([layers[2]], 0, .1, dtype=tf.float32), name='b2')
        #b3 = tf.Variable(tf.random_normal([layers[3]], 0, .1, dtype=tf.float32), name='b3')
        #b4 = tf.Variable(tf.random_normal([layers[4]], 0, .1, dtype=tf.float32), name='b4')
        #b_out = tf.Variable(tf.random_normal([layers[5]], 0, .1, dtype=tf.float32), name='b_out')
        # Actication function
        #act = tf.nn.sigmoid
        act = tf.nn.relu

    with tf.name_scope('TRAIN'):
        learning_rate = .5
        y_out = model(inputs, [m1, m2, m3], act)
        
        cost_op = tf.reduce_mean(tf.pow(y_out - outputs, 2))
        train_op = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost_op)

    # Actual training
    epoch, last_cost, max_epochs, tolerance = 0, 0, 5000, 1e-6

    print 'Begin training...'
    sess = tf.Session()
    with sess.as_default():
        init = tf.global_variables_initializer()
        sess.run(init)

        costs = []
        epochs = []
        
        train_x = norm2byte(train_x).astype(np.float32)
        train_y = norm2byte(train_y).astype(np.float32)

        while True:
            sess.run(train_op, feed_dict={inputs: train_x, outputs: train_y})
            if epoch % 1000 == 0:
                cost = sess.run(cost_op, feed_dict={inputs: train_x, outputs: train_y})
                costs.append(cost)
                epochs.append(epoch)

                print 'Epoch: {} -- Error: {}'.format(epoch, cost)

                if epoch > max_epochs:
                    print 'Max # of iteration reached, stop.'
                    break
                last_cost = cost
            epoch += 1

        # Gen test sets
        test_x = test_x[:args.N]
        test_y = test_y[:args.N]

        # Quantize inputs/outputs/weights
        qtz_input = norm2byte(test_x)
        qtz_output = norm2byte(test_y)
        m1_val = sess.run(m1)
        qtz_m1 = norm2byte(m1_val)
        m2_val = sess.run(m2)
        qtz_m2 = norm2byte(m2_val)
        m3_val = sess.run(m3)
        qtz_m3 = norm2byte(m3_val)

        # Pad/Save inputs/weights
        HW_WIDTH = 16

        # input
        shape = qtz_input.shape
        pad_input = np.zeros((shape[0], HW_WIDTH), dtype=np.int8)
        pad_input[:shape[0], :shape[1]] = qtz_input
        print 'padded input: {}'.format(pad_input)
        np.save(args.save_input_path, pad_input)

        # weights
        shape = qtz_m1.shape
        pad_m1 = np.zeros((HW_WIDTH, HW_WIDTH), dtype=np.int8)
        pad_m1[:shape[0], :shape[1]] = qtz_m1
        pad_m1.reshape((1, HW_WIDTH, HW_WIDTH))
        print 'padded m1: {}'.format(pad_m1)

        shape = qtz_m2.shape
        pad_m2 = np.zeros((HW_WIDTH, HW_WIDTH), dtype=np.int8)
        pad_m2[:shape[0], :shape[1]] = qtz_m2
        pad_m2.reshape((1, HW_WIDTH, HW_WIDTH))
        print 'padded m2: {}'.format(pad_m2)

        shape = qtz_m3.shape
        pad_m3 = np.zeros((HW_WIDTH, HW_WIDTH), dtype=np.int8)
        pad_m3[:shape[0], :shape[1]] = qtz_m3
        pad_m3.reshape((1, HW_WIDTH, HW_WIDTH))
        print 'padded m3: {}'.format(pad_m3)
        padded_weights = np.array((pad_m1, pad_m2, pad_m3))
        print 'padded weights: {}'.format(padded_weights)
        np.save(args.save_weight_path, padded_weights)

        # Update weights
        sess.run(m1.assign(qtz_m1))
        sess.run(m2.assign(qtz_m2))
        sess.run(m3.assign(qtz_m3))

        # Test with 8b inputs/weights
        pred_y = (sess.run(y_out, feed_dict={inputs: qtz_input, outputs: qtz_output})).astype(np.int8)
        np.save(args.save_output_path, pred_y)
        print 'Prediction\nReal\tPredicted'
        for (y, y_hat) in zip(test_y, pred_y):
            print '{}\t{}'.format(y, y_hat)
        
        r2 = metrics.r2_score(test_y, pred_y)
        print 'R2: {}'.format(r2)

def norm2byte(mat, shape=None):
    pos_mat = np.vectorize(lambda x: np.abs(x))(mat)
    max_w = np.amax(pos_mat)
    mat = np.vectorize(lambda x: (127 * x/max_w).astype(np.int8))(mat)
    return mat.reshape(shape) if shape else mat

def parse_args():
    global args

    parser = argparse.ArgumentParser()

    parser.add_argument('--save-weight-path', action='store',
                        help='path to save weights.')
    parser.add_argument('--save-input-path', action='store',
                        help='path to save inputs.')
    parser.add_argument('--save-output-path', action='store',
                        help='path to save predicts.')
    parser.add_argument('--N', action='store', type=int,
                        help='number of test cases.')
    args = parser.parse_args()

if __name__ == '__main__':
    parse_args()
    main()
