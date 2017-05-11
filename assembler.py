"""
===assembly====

Instruction has the following format:
    INST: OP, SRC, TAR, LEN (or FUNC), FLAG
    LENGTH: 1B, VAR, VAR, 1B, 1B

OPCODE may define flags by using dot (.) separator following
opcode string.

For ACT instruction, function byte is defined using the following
mapping:
    0x0 -> ReLU
    0x1 -> Sigmoid
    0x2 -> MaxPooling

Comments start with #.

EXAMPLES:
    # example program
    RHM 1, 2, 3 # first instruction
    WHM 1, 2, 3
    RW 0xab
    MMC 100, 2, 3
    MMC.C 100, 2, 3
    ACT 0xab, 12, 1
    NOP
    HLT

===binary encoding====

INST is encoded in a little-endian format.
OPCODE values are defined in OPCODE2BIN.
FLAG field is r|r|r|r|r|o|s|c, r stands for reserved bit, s for switch bit,
c for convolve bit, and o for override bit.

SRC and TAR are addresses. They can be of variable length defined in
global dict OPCODE2BIN.

SRC/TAR takes 5B for memory operations to support at least 8GB addressing,
3B for Unified Buffer addressing (96KB), 2B for accumulator buffer addressing
(4K).

"""

import argparse
import re
from isa import *

args = None

TOP_LEVEL_SEP = re.compile(r'[a-zA-Z]+\s+')

SUFFIX = '.out'


def DEBUG(string):
    if args.debug:
        print(string)
    else:
        return


def assemble(path, n):
    """ Translates an assembly code file into a binary.
    """

    assert path
    with open(path, 'r') as code:
        lines = code.readlines()
    code.close()
    n = len(lines) if not n else n
    write_path = path[:path.rfind('.')] if path.rfind('.') > -1 else path
    bin_code = open(write_path + SUFFIX, 'wb')
    counter = 0
    for line in lines:
        line = line.partition('#')[0]
        if not line:
            continue
        counter += 1
        operands = TOP_LEVEL_SEP.split(line)[1]
        operands = [int(op.strip(), 0) for op in operands.split(',')] if operands else []
        opcode = line.split()[0].strip()
        assert opcode
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
        if 'O' in flags:
            flag |= OVERWRITE_MASK
        if 'Q' in flags:
            flag |= FUNC_SIGMOID_MASK
        if 'R' in flags:
            flag |= FUNC_RELU_MASK

        # binary for flags
        bin_flags = flag.to_bytes(1, byteorder=ENDIANNESS)

        opcode, n_src, n_tar, n_3rd = OPCODE2BIN[opcode]

        # binary representation for opcode
        bin_opcode = opcode.to_bytes(1, byteorder=ENDIANNESS)

        # binary for oprands
        bin_operands = b''
        if len(operands) == 0:
            bin_operands = b''
        elif len(operands) == 1:
            bin_operands = operands[0].to_bytes(n_src, byteorder=ENDIANNESS)
        elif len(operands) == 3:
            bin_operands += operands[0].to_bytes(n_src, byteorder=ENDIANNESS)
            bin_operands += operands[1].to_bytes(n_tar, byteorder=ENDIANNESS)
            bin_operands += operands[2].to_bytes(n_3rd, byteorder=ENDIANNESS)

        # binary for instruction
        # bin_rep = bin_opcode + bin_operands + bin_flags
        bin_rep = bin_flags + bin_operands + bin_opcode

        if len(bin_rep) < INSTRUCTION_WIDTH_BYTES:
            x = 0
            # bin_rep += x.to_bytes(14 - len(bin_rep), byteorder=ENDIANNESS)
            zeros = x.to_bytes(INSTRUCTION_WIDTH_BYTES - len(bin_rep), byteorder=ENDIANNESS)
            # bin_rep = bin_opcode + bin_operands + zeros + bin_flags
            bin_rep = bin_flags + bin_operands + zeros + bin_opcode

        DEBUG(line[:-1])
        DEBUG(bin_rep)

        # write to file
        bin_code.write(bin_rep)

        if counter == n:
            break
    bin_code.close()


def parse_args():
    global args

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', action='store',
                        help='path to source file.')
    parser.add_argument('--n', action='store', type=int, default=0,
                        help='only parse first n lines of code, for dbg only.')
    parser.add_argument('--debug', action='store_true',
                        help='switch debug prints.')
    args = parser.parse_args()


if __name__ == '__main__':
    parse_args()
    assemble(args.path, args.n)
