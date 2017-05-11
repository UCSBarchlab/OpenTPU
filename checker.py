""" The checker assumes results are always written to hostmem
    consecutively starting from location 0.

    If the result is shorter than HW width: (X is don't care)

    --------HW WIDTH---------
    D D D D D D D X X X X X X 
    D D D D D D D X X X X X X

    else:

    --------HW WIDTH---------
    D D D D D D D D D D D D D
    D D X X X X X X X X X X X
    D D D D D D D D D D D D D
    D D X X X X X X X X X X X

"""

import argparse
import numpy as np

args = None

def equal(a1, a2):
    assert a1.shape == a2.shape, 'result file shape mismatch.'
    for x, y in np.nditer([a1, a2]):
        assert x == y, 'result value mismatch.'

def check(p1, p2, width=None):
    r1 = np.load(p1)
    r2 = np.load(p2)
    if not width:
        # Checking sim8 against hw8.
        equal(r1, r2)
    else:
        # Checking gt32 against sim32.
        #assert width == r2.shape[1]
        r_width = r1.shape[1]
        if r_width <= width:
            r2 = r2[:, :r_width]
            equal(r1, r2)
        else:
            r2 = np.concatenate((r2[::2], r2[1::2]), axis=1)
            r2 = r2[:, :r_width]
            equal(r1, r2)


def parse_args():
    global args

    parser = argparse.ArgumentParser()

    parser.add_argument('--width', action='store', type=int, default=16,
                        help='HW WIDTH.')
    parser.add_argument('--gt32', action='store', default='gt32.npy',
                        help='path to f32 ground truth result.')
    parser.add_argument('--sim32', action='store', default='sim32.npy',
                        help='path to f32 simulator result.')
    parser.add_argument('--sim8', action='store', default='sim8.npy',
                        help='path to i8 simulator result.')
    parser.add_argument('--hw8', action='store', default='hw8.npy',
                        help='path to i8 hardware result.')
    args = parser.parse_args()


if __name__ == '__main__':
    parse_args()
    print 'HW width set to %d.' % args.width
    check(args.gt32, args.sim32, args.width)
    print '32-bit passed.'
    check(args.sim8, args.hw8)
    print '8-bit passed.'
