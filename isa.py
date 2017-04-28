# Map text opcode to instruction decomposition info.
# Str -> (opcode_value, src_len, tar_len, 3rd_len)

HOST_ADDR_SIZE = 8 # 64-bit addressing
DRAM_ADDR_SIZE = 5 # 33-bit addressing (TPU has 8 GB on-chip DRAM)
UB_ADDR_SIZE = 3 # 17-bit addressing for Unified Buffer
ACC_ADDR_SIZE = 2 # 12-bit addressing for accumulator

OPCODE2BIN = {
        'RHM':  (0x0, HOST_ADDR_SIZE, UB_ADDR_SIZE,   1),
        'WHM':  (0x1, UB_ADDR_SIZE,   HOST_ADDR_SIZE, 1),
        'RW':   (0x2, DRAM_ADDR_SIZE, 0,              0),
        'MMC':  (0x3, UB_ADDR_SIZE,   ACC_ADDR_SIZE,  2),
        'ACT':  (0x4, ACC_ADDR_SIZE,  UB_ADDR_SIZE,   1),
        'SYNC': (0x5, 0, 0, 0),
        'NOP':  (0x6, 0, 0, 0),
        'HLT':  (0x7, 0, 0, 0),
        }

BIN2OPCODE = {v[0]: k for k, v in OPCODE2BIN.items()}

SWITCH_MASK = 0x1
CONV_MASK = 0x2
OVERWRITE_MASK = 0x4 # whether MMC should overwrite accumulator value or add to it
