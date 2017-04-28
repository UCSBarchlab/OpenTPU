import argparse
import numpy as np

args = None

def gen_mem(path, shape):
    mem = np.random.rand(*shape)
    np.save(path, mem)

def parse_args():
    global args

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', action='store',
                        help='path to source file.')
    parser.add_argument('--shape', action='store', type=int, nargs='+',
                        help = 'shape of matrix to generate.')
    parser.add_argument('--debug', action='store_true',
                        help='switch debug prints.')
    args = parser.parse_args()


if __name__ == '__main__':
    parse_args()
    gen_mem(args.path, args.shape)
