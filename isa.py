# Map text opcode to instruction decomposition info.
# Str -> (opcode_value, src_len, tar_len, 3rd_len)
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

BIN2OPCODE = {v[0]: k for k, v in OPCODE2BIN.items()}

SWITCH_MASK = 0x1
CONV_MASK = 0x2
