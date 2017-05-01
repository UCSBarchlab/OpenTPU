# Map text opcode to instruction decomposition info.
# Str -> (opcode_value, src_len, tar_len, 3rd_len)

"""
===binary encoding====

INST is encoded in a little-endian format.
OPCODE values are defined in OPCODE2BIN.
FLAG field is r|r|f|f|f|o|s|c, r stands for reserved bit, s for switch bit,
c for convolve bit, f for function select bits, and o for override bit.

SRC and TAR are addresses. They can be of variable length defined in
global dict OPCODE2BIN.

SRC/TAR takes 5B for memory operations to support at least 8GB addressing,
3B for Unified Buffer addressing (96KB), 2B for accumulator buffer addressing
(4K).

"""

HOST_ADDR_SIZE = 8 # 64-bit addressing
DRAM_ADDR_SIZE = 5 # 33-bit addressing (TPU has 8 GB on-chip DRAM)
UB_ADDR_SIZE = 3 # 17-bit addressing for Unified Buffer
ACC_ADDR_SIZE = 2 # 12-bit addressing for accumulator

OPCODE2BIN = {
        'NOP':  (0x0, 0, 0, 0),
        'WHM':  (0x1, UB_ADDR_SIZE,   HOST_ADDR_SIZE, 1),
        'RW':   (0x2, DRAM_ADDR_SIZE, 0,              0),
        'MMC':  (0x3, UB_ADDR_SIZE,   ACC_ADDR_SIZE,  2),
        'ACT':  (0x4, ACC_ADDR_SIZE,  UB_ADDR_SIZE,   1),
        'SYNC': (0x5, 0, 0, 0),
        'RHM':  (0x6, HOST_ADDR_SIZE, UB_ADDR_SIZE,   1),
        'HLT':  (0x7, 0, 0, 0),
        }

BIN2OPCODE = {v[0]: k for k, v in OPCODE2BIN.items()}

SWITCH_MASK = 0x1
CONV_MASK = 0x2
OVERWRITE_MASK = 0x4 # whether MMC should overwrite accumulator value or add to it
FUNC_SIGMOID_MASK = 0b001 << 3
FUNC_RELU_MASK = 0b010 << 3
