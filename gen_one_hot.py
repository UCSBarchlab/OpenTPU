import argparse
import numpy as np
from gen_mem import gen_mem

args = None

def gen_one_hot(lower=-5, upper=5, shape=(8, 8)):
    #one_hot = np.random.randint(-5, 5, (8, 8), dtype=np.int8)
    one_hot = np.random.randint(lower, upper, shape, dtype=np.int8)
    # We eigher generate a squre matrix for training or generate a vector for testing.
    if shape[0] == shape[1]:
        for i in xrange(shape[0]):
            one_hot[i, i] = 64
    else:
        assert shape[1] == 1
        for i in xrange(shape[0]):
            one_hot[i, 0] = np.random.randint(lower, upper, dtype=np.int8)
    return one_hot

def gen_nn(path, shape, lower=None, upper=None):
    #nn = np.random.randint(lower, upper, shape, dtype=np.int8)
    nn = gen_one_hot(lower, upper, shape)
    print(nn)
    gen_mem(path, nn)

def parse_args():
    global args

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', action='store',
                        help='dest file.')
    parser.add_argument('--shape', action='store', type=int, nargs='+',
                        help = 'shape of matrix to generate.')
    parser.add_argument('--debug', action='store_true',
                        help='switch debug prints.')
    parser.add_argument('--range', type=int, nargs=2,
                        help='gen rand in [lower, upper)')
    args = parser.parse_args()


if __name__ == '__main__':
    parse_args()
    gen_nn(args.path, args.shape, *args.range)
