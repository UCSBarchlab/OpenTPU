"""
The assembly format for most instructions (RHM, WHM, MMC, ACT) is
    INSTRUCTION SRC, DEST, LENGTH 
For RW, it is
    RW SRC
for HLT, it is
    HLT

=== Binary Encoding ====

| opcode | flags | length | addr | ub addr |
|   1    |   1   |   1    |  8   |    3    |
|13    13|12   12|11    11|10   3|2       0|

All numbers above are expressed in BYTES.
The 'addr' field is used for host memory address (for RHM and WHM),
weight DRAM address (for RW), and accumulator address (for MMC and ACT).
For the later two, the field is larger than necessary, and only the lower bits are used.
'ub addr' is always a Unified Buffer address.
'length' is the number of vectors to read/write/process.

FLAG field is r|r|f|f|f|o|s|c, r stands for reserved bit, s for switch bit,
c for convolve bit, f for function select bits, and o for override bit.

"""

# ENDIANNESS = 'big'
ENDIANNESS = 'little'

INSTRUCTION_WIDTH_BYTES = 14

HOST_ADDR_SIZE = 8 # 64-bit addressing
DRAM_ADDR_SIZE = 5 # 33-bit addressing (TPU has 8 GB on-chip DRAM)
UB_ADDR_SIZE = 3 # 17-bit addressing for Unified Buffer
ACC_ADDR_SIZE = 2 # 12-bit addressing for accumulator
OP_SIZE = 1
FLAGS_SIZE = 1
ADDR_SIZE = 8
UB_ADDR_SIZE = 3
LEN_SIZE = 1

UBADDR_START = 0
UBADDR_END = 3
ADDR_START = 3
ADDR_END = 11
LEN_START = 11
LEN_END = 12
FLAGS_START = 12
FLAGS_END = 13
OP_START = 13
OP_END = 14

# Map text opcode to instruction decomposition info.
# Str -> (opcode_value, src_len, dst_len, 3rd_len)
OPCODE2BIN = {
        'NOP':  (0x0, 0, 0, 0),
        'WHM':  (0x1, UB_ADDR_SIZE,   HOST_ADDR_SIZE, 1),
        'RW':   (0x2, DRAM_ADDR_SIZE, 0,              1),
        'MMC':  (0x3, UB_ADDR_SIZE,   ACC_ADDR_SIZE,  1),
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
