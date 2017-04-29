import argparse
import numpy as np

args = None

def gen_one_hot():
    one_hot = np.zeros((8, 8))
    for i in xrange(8):
        one_hot[i, i] = np.int8(1)
    return one_hot

def gen_nn(path, shape, lower=None, upper=None):
    nn = np.random.randint(lower, upper, shape, dtype=np.int8)
    nn = gen_one_hot()
    print(nn)
    np.save(path, nn)

def parse_args():
    global args

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', action='store',
                        help='path to source file.')
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
