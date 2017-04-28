#!/usr/bin/env python

import argparse
import re

args = None

OPCODE2BIN = {
        'RHM': (0x0, 5, 3, 1),
        'WHM': (0x1, 5, 3, 1),
        'RW': (0x2, 5, 0, 0),
        'MMC': (0x3, 3, 2, 2),
        'ACT': (0x4, 2, 3, 1),
        'SYNC': (0x5, 0, 0, 0),
        'NOP': (0x6, 0, 0, 0),
        'HLT': (0x7, 0, 0, 0),
        }

SWITCH_MASK = 0x1
CONV_MASK = 0x2

TOP_LEVEL_SEP = re.compile(r'[a-zA-Z]+\s+')

def assemble(path):
    """ Translates an assembly code file into a binary.
    """

    assert path
    with open(path, 'r') as code:
        lines = code.readlines()
    code.close()

    bin_code = open(path+'.a', 'wb')
    for line in lines:
        oprands = TOP_LEVEL_SEP.split(line)[1]
        oprands = [int(op.strip(), 0) for op in oprands.split(',')] if oprands else []
        opcode = line.split()[0]
        comps = opcode.split('.')
        assert comps and len(comps) < 3
        if len(comps) == 1:
            opcode = comps[0]
            flags = ''
        else:
            opcode = comps[0]
            flags = comps[1]

        flag = 0
        if 'S' in flags:
            flag |= SWITCH_MASK
        if 'C' in flags:
            flag |= CONV_MASK

        # python3
        bin_flags = flag.to_bytes(1, byteorder='little')

        opcode, n_src, n_tar, n_3rd = OPCODE2BIN[opcode]

        # binary representation for opcode.
        bin_opcode = opcode.to_bytes(1, byteorder='little')

        bin_oprands = b''
        if len(oprands) == 0:
            bin_oprands = b''
        elif len(oprands) == 1:
            bin_oprands = oprands[0].to_bytes(n_src, byteorder='little')
        elif len(oprands) == 3:
            bin_oprands += oprands[0].to_bytes(n_src, byteorder='little')
            bin_oprands += oprands[1].to_bytes(n_tar, byteorder='little')
            bin_oprands += oprands[2].to_bytes(n_3rd, byteorder='little')

        bin_rep = bin_opcode + bin_oprands + bin_flags
        bin_code.write(bin_rep)
    bin_code.close()

def parse_args():
    global args

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', action='store',
	    help='path to source file.')

    args = parser.parse_args()

if __name__ == '__main__':
    parse_args()
    assemble(args.path)

