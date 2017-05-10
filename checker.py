import argparse
import numpy as np

args = None

def check(p1, p2):
    r1 = np.load(p1)
    r2 = np.load(p2)
    assert r1.shape == r2.shape, 'result file shape mismatch.'
    for x, y in np.nditer([r1, r2]):
        assert x == y, 'result value mismatch.'

def parse_args():
    global args

    parser = argparse.ArgumentParser()

    parser.add_argument('--gt32', action='store', default='gt32.npy',
                        help='path to 32-bit ground truth result.')
    parser.add_argument('--sim32', action='store', default='sim32.npy',
                        help='path to 32-bit simulator result.')
    parser.add_argument('--sim8', action='store', default='sim8.npy',
                        help='path to 8-bit simulator result.')
    parser.add_argument('--hw8', action='store', default='hw8.npy',
                        help='path to 8-bit hardware result.')
    args = parser.parse_args()


if __name__ == '__main__':
    parse_args()
    check(args.gt32, args.sim32)
    print '32-bit passed.'
    check(args.sim8, args.hw8)
    print '8-bit passed.'
