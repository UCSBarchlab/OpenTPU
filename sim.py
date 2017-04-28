# coding=utf-8
import sys
import numpy as np
import isa


class TPUSim(object):
    def __init__(self, program_filename, dram_filename, hostmem_filename):
        self.program = open(program_filename, 'rb')
        self.weight_memory = np.load(dram_filename)
        self.host_memory = np.load(hostmem_filename)
        self.unified_buffer = np.zeros((96000, 256))
        self.accumulator = np.zeros((4000, 256))
        self.weight_fifo = []

    def run(self):
        # load program and execute instructions
        while True:
            instruction = self.decode()
            opcode, operands = instruction[0], instruction[1:]
            if opcode in ['RHM', 'WHM', 'RW']:
                self.memops(opcode, *operands)
            elif opcode == 'MMC':
                # TODO: unpack like a decen tpernson
                self.matrix_multiply_convolve(*operands)
            elif opcode == 'ACT':
                self.act(*operands)
            elif opcode == 'SYNC':
                self.sync(*operands)
            elif opcode == 'NOP':
                print('No operation')
            elif opcode == 'HLT':
                print('H A L T')
                break
            else:
                raise Exception('WAT (╯°□°）╯︵ ┻━┻')

        # all done, exit
        np.save('unified_buffer.npy', self.unified_buffer)
        self.program.close()

    def decode(self):
        opcode = int.from_bytes(self.program.read(1), byteorder='little')
        opcode = isa.BIN2OPCODE[opcode]

        src_addr = int.from_bytes(self.program.read(isa.OPCODE2BIN[opcode][1]), byteorder='little')
        dest_addr = int.from_bytes(self.program.read(isa.OPCODE2BIN[opcode][2]), byteorder='little')
        length = int.from_bytes(self.program.read(isa.OPCODE2BIN[opcode][3]), byteorder='little')
        flag = int.from_bytes(self.program.read(1), byteorder='little')
        return opcode, src_addr, dest_addr, length, flag

    # opcodes
    def sync(self, flags):
        print('sink')

    def act(self, src, dest, length, flag):
        print('ACTIVATE! (or pool?)')

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
        out = np.matmul(inp, self.weight_fifo.pop(0))
        overwrite = isa.OVERWRITE_MASK | flags
        if overwrite:
            self.accumulator[accum_addr:accum_addr + size] = out
        else:
            self.accumulator[accum_addr:accum_addr + size] += out


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage:', sys.argv[0], 'PROGRAM_BINARY DRAM_FILE HOST_FILE')
        sys.exit(0)

    tpusim = TPUSim(sys.argv[1], sys.argv[2], sys.argv[3])
    tpusim.run()
