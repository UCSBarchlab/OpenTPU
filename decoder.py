from pyrtl import *
import isa

DATASIZE = 8
MATSIZE = 16
ACCSIZE = 8

def decode(instruction):
    """
    :param instruction: instruction + optional operands + flags
    """

    accum_raddr = WireVector(ACCSIZE)
    accum_waddr = WireVector(ACCSIZE)
    accum_overwrite = WireVector(1)
    switch_weights = WireVector(1)
    weights_we = WireVector(1)

    ub_addr = WireVector(24)  # goes to FSM
    ub_raddr = WireVector(isa.UB_ADDR_SIZE * 8)  # goes to UB read addr port
    ub_waddr = WireVector(isa.UB_ADDR_SIZE * 8)

    whm_length = WireVector(8)
    rhm_length = WireVector(8)
    mmc_length = WireVector(16)
    act_length = WireVector(8)

    rhm_addr = WireVector(isa.HOST_ADDR_SIZE * 8)
    whm_addr = WireVector(isa.HOST_ADDR_SIZE * 8)

    dispatch_mm = WireVector(1)
    dispatch_act = WireVector(1)
    dispatch_rhm = WireVector(1)
    dispatch_whm = WireVector(1)
    
    op = instruction[:8]

    with conditional_assignment:
        with op == isa.OPCODE2BIN['NOP'][0]:
            pass
        with op == isa.OPCODE2BIN['WHM'][0]:
            whm_addr_start = 8+(isa.UB_ADDR_SIZE * 8)
            whm_addr_end = whm_addr_start + isa.HOST_ADDR_SIZE * 8
            ub_raddr |= instruction[8:whm_addr_start]
            whm_addr |= instruction[whm_addr_start:whm_addr_end]
            dispatch_whm |= 1
            whm_length |= instruction[-16:-8]
        with op == isa.OPCODE2BIN['RW'][0]:
            weights_we |= 1
        with op == isa.OPCODE2BIN['MMC'][0]:
            dispatch_mm |= 1
            ub_addr |= instruction[8:8+(isa.UB_ADDR_SIZE * 8)]
            accum_waddr_start = 8 + isa.UB_ADDR_SIZE
            accum_waddr_end = 8+(isa.UB_ADDR_SIZE * 8) + (isa.ACC_ADDR_SIZE * 8)
            accum_waddr |= instruction[accum_waddr_start:accum_waddr_end]
            mmc_length |= instruction[accum_waddr_end:accum_waddr_end + 8]
            flags = instruction[-8:]
            accum_overwrite |= flags & isa.OVERWRITE_MASK
            switch_weights |= flags & isa.SWITCH_MASK
            # TODO: MMC may deal with convolution, set/clear that flag
        with op == isa.OPCODE2BIN['ACT'][0]:
            dispatch_act |= 1
            opcode_end = 8
            acc_addr_end = opcode_end + isa.ACC_ADDR_SIZE * 8
            ub_addr_end = acc_addr_end + isa.UB_ADDR_SIZE * 8

            accum_raddr |= instruction[opcode_end:acc_addr_end]
            ub_waddr |= instruction[acc_addr_end:ub_addr_end]
            act_length |= instruction[-16:-8]
            # TODO: ACT takes function select bits
        with op == isa.OPCODE2BIN['SYNC'][0]:
            pass
        with op == isa.OPCODE2BIN['RHM'][0]:
            rhm_addr_start = 8
            dispatch_rhm |= 1
            ub_addr_start = 8 + isa.HOST_ADDR_SIZE * 8
            ub_addr_end = ub_addr_start + isa.UB_ADDR_SIZE * 8
            rhm_addr |= instruction[rhm_addr_start:ub_addr_start]
            ub_raddr |= instruction[ub_addr_start:ub_addr_end]
            rhm_length |= instruction[-16:-8]
        with op == isa.OPCODE2BIN['HLT'][0]:
            pass

        with otherwise:
            print("otherwise")

    return dispatch_mm, dispatch_act, dispatch_rhm, dispatch_whm, ub_addr, ub_raddr, ub_waddr, rhm_addr, whm_addr, rhm_length, whm_length, mmc_length, act_length, accum_raddr, accum_waddr, accum_overwrite, switch_weights, weights_we

