# coding=utf-8
import argparse
import sys
import numpy as np
from collections import deque
from math import exp

import isa

args = None
# width of the tile
WIDTH = 8


class TPUSim(object):
    def __init__(self, program_filename, dram_filename, hostmem_filename):
        # TODO: switch b/w 32-bit float vs int
        self.program = open(program_filename, 'rb')
        self.weight_memory = np.load(dram_filename)
        self.host_memory = np.load(hostmem_filename)
        if not args.raw:
            assert self.weight_memory.dtype == np.int8, 'DRAM weight mem is not 8-bit ints'
            assert self.host_memory.dtype == np.int8, 'Hostmem not 8-bit ints'
        self.unified_buffer = (np.zeros((96000, WIDTH), dtype=np.float32) if args.raw else
            np.zeros((96000, WIDTH), dtype=np.int8))
        self.accumulator = (np.zeros((4000, WIDTH), dtype=np.float32) if args.raw else
            np.zeros((4000, WIDTH), dtype=np.int32))
        self.weight_fifo = deque()

    def run(self):
        # load program and execute instructions
        while True:
            instruction = self.decode()
            opcode, operands = instruction[0], instruction[1:]
            if opcode in ['RHM', 'WHM', 'RW']:
                self.memops(opcode, *operands)
            elif opcode == 'MMC':
                self.matrix_multiply_convolve(*operands)
            elif opcode == 'ACT':
                self.act(*operands)
            elif opcode == 'SYNC':
                pass
            elif opcode == 'NOP':
                pass
            elif opcode == 'HLT':
                print('H A L T')
                break
            else:
                raise Exception('WAT (╯°□°）╯︵ ┻━┻')

        # all done, exit
        savepath = 'sim32.npy' if args.raw else 'sim8.npy'
        np.save(savepath, self.host_memory)
        print(self.host_memory)
        self.program.close()

        print("""ALL DONE!
        (•_•)
        ( •_•)>⌐■-■
        (⌐■_■)""")

    def decode(self):
        opcode = int.from_bytes(self.program.read(isa.OP_SIZE), byteorder='big')
        opcode = isa.BIN2OPCODE[opcode]

        flag = int.from_bytes(self.program.read(isa.FLAGS_SIZE), byteorder='big')
        length = int.from_bytes(self.program.read(isa.LEN_SIZE), byteorder='big')
        src_addr = int.from_bytes(self.program.read(isa.ADDR_SIZE), byteorder='big')
        dest_addr = int.from_bytes(self.program.read(isa.UB_ADDR_SIZE), byteorder='big')
        #print('{} decoding: len {}, flags {}, src {}, dst {}'.format(opcode, length, flag, src_addr, dest_addr))
        return opcode, src_addr, dest_addr, length, flag

    # opcodes
    def act(self, src, dest, length, flag):
        print('ACTIVATE!')

        result = self.accumulator[src:src+length]
        if flag & isa.FUNC_RELU_MASK:
            print('  RELU!!!!')
            vfunc = np.vectorize(lambda x: 0 * x if x < 0. else x)
        elif flag & isa.FUNC_SIGMOID_MASK:
            print('  SIGMOID')
            vfunc = np.vectorize(lambda x: int(255./(1.+exp(-x))))
        else:
            vfunc = np.vectorize(lambda x: x)
            #raise Exception('(╯°□°）╯︵ ┻━┻ bad activation function!')

        result = vfunc(result)

        # downsample/normalize if needed
        if not args.raw:
            result = [v & 0x000000FF for v in result]
        self.unified_buffer[dest:dest+length] = result

    def memops(self, opcode, src_addr, dest_addr, length, flag):
        print('Memory xfer! host: {} unified buffer: {}: length: {} (FLAGS? {})'.format(
            src_addr, dest_addr, length, flag
        ))

        if opcode == 'RHM':
            print('  read host memory to unified buffer')
            self.unified_buffer[dest_addr:dest_addr + length] = self.host_memory[src_addr:src_addr + length]
        elif opcode == 'WHM':
            print('  write unified buffer to host memory')
            self.host_memory[dest_addr:dest_addr + length] = self.unified_buffer[src_addr:src_addr + length]
        elif opcode == 'RW':
            print('  read weights from DRAM into MMU')
            self.weight_fifo.append(self.weight_memory[src_addr])
        else:
            raise Exception('WAT (╯°□°）╯︵ ┻━┻')

    def matrix_multiply_convolve(self, ub_addr, accum_addr, size, flags):
        print('Matrix things....')
        print('  UB@{} + {} -> MMU -> accumulator@{} + {}'.format(
            ub_addr, size, accum_addr, size
        ))

        inp = self.unified_buffer[ub_addr: ub_addr + size]
        print('MMC input shape: {}'.format(inp.shape))
        weight_mat = self.weight_fifo.popleft()
        print('MMC weight: {}'.format(weight_mat))
        out = np.matmul(inp, weight_mat)
        print('MMC output shape: {}'.format(out.shape))
        overwrite = isa.OVERWRITE_MASK & flags
        if overwrite:
            self.accumulator[accum_addr:accum_addr + size] = out
        else:
            self.accumulator[accum_addr:accum_addr + size] += out

def parse_args():
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument('program', action='store',
                        help='Path to assembly program file.')
    parser.add_argument('dram_file', action='store',
                        help='Path to dram file.')
    parser.add_argument('host_file', action='store',
                        help='Path to host file.')
    parser.add_argument('--raw', action='store_true', default=False,
                        help='Gen sim32.npy instead of sim8.npy.')
    args = parser.parse_args()

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage:', sys.argv[0], 'PROGRAM_BINARY DRAM_FILE HOST_FILE')
        sys.exit(0)
    
    parse_args()
    tpusim = TPUSim(args.program, args.dram_file, args.host_file)
    tpusim.run()
