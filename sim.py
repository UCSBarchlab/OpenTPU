import sys
import numpy as np


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
        self.program.close()

    # opcodes
    def read_host_mem(self, host_addr, ub_addr, size):
        """
        Reads a range of memory addresses from "start_addr" to "stop_addr" in the
        host_memory
        """
        self.unified_buffer[ub_addr:ub_addr + size] = self.host_memory[host_addr:host_addr + size]
        print("Host Memory Read Complete")

    def write_host_mem(self, ub_addr, host_addr, size):
        """
        Writes "data" to a range of memory addresses from "start_addr" to
        "stop_addr" in host_memory
        """
        self.host_memory[host_addr:host_addr + size] = self.unified_buffer[ub_addr:ub_addr + size]
        print("Host Memory Write Complete")

    def read_weights(self, weight_dram_addr):
        """
        Moves one tiles into the weight fifo
        """
        self.weight_fifo.append(self.weight_memory[weight_dram_addr])

    def matrix_multiply_convolve(self, switch, ub_addr, accumulator_addr, size, weight_fifo):
        inp = self.unified_buffer[ub_addr: ub_addr + size]
        out = np.matmul(inp, weight_fifo.pop())
        self.accumulator[accumulator_addr:accumulator_addr + 256] = out  # because matmul unit is 256*256


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'Usage:', sys.argv[0], 'PROGRAM_BINARY DRAM_FILE HOST_FILE'
        sys.exit(0)

    tpusim = TPUSim(sys.argv[1], sys.argv[2], sys.argv[3])
    tpusim.run()
