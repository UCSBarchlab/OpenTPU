import argparse
import numpy as np
import pickle
from gen_mem import gen_mem

# train code from http://iamtrask.github.io/2015/07/12/basic-python-network/

# sigmoid function
def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def relu(X):
    map_func = np.vectorize(lambda x: max(x, 0))
    return map_func(X)

def train(path, lpath):    
    X = np.load(path)
    y = np.load(lpath)

    print 'X: {}'.format(X)
    print 'y: {}'.format(y)
    # seed random numbers to make calculation
    # deterministic (just a good practice)
    np.random.seed(1)

    # initialize weights randomly with mean 0
    syn0 = np.random.random((X.shape[-1], y.shape[-1]))

    nonlin = sigmoid

    for iter in xrange(10000):

        # forward propagation
        l0 = X
        l1 = nonlin(np.dot(l0,syn0))

        # how much did we miss?
        l1_error = y - l1

        # multiply how much we missed by the 
        # slope of the sigmoid at the values in l1
        l1_delta = l1_error * nonlin(l1)

        # update weights
        syn0 += np.dot(l0.T,l1_delta)
    print 'syn0: {}'.format(syn0)
    print 'l1: {}'.format(l1)
    with open('simple_nn_gt', 'w') as f:
        pickle.dump((l1, syn0), f)
        f.close()
    
    syn0 = float2byte(syn0)
    gen_mem('simple_nn_weight_dram', syn0)

args = None

def float2byte(mat):
    pos_mat = np.vectorize(lambda x: np.abs(x))(mat)
    max_w = np.amax(pos_mat)
    mat = np.vectorize(lambda x: (127 * x/max_w).astype(np.int8))(mat)
    return mat.reshape(1, 8, 8)

def parse_args():
    global args

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', action='store',
                        help='path to dataset file.')
    parser.add_argument('--label', action='store',
                        help='path to the label file.')
    parser.add_argument('--debug', action='store_true',
                        help='switch debug prints.')
    args = parser.parse_args()


if __name__ == '__main__':
    parse_args()
    train(args.path, args.label)
