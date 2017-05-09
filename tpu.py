from pyrtl import *

import isa
from isa import INSTRUCTION_WIDTH
from decoder import decode
from matrix import MMU_top
from act_top import act_top2

############################################################
#  Machine Parameters
############################################################

DWIDTH = 8  # 8
MATSIZE = 4  # 256
#ACCUMSIZE = 8  # 13 (4k entries)
ACCUMSIZE = isa.ACC_ADDR_SIZE * 8
#UBSIZE = 15  #  (96k entries in original; need one 2^16 and one 2^15 memory for that)
UBSIZE = isa.UB_ADDR_SIZE * 8
IMEM_SIZE = 15

############################################################
#  Control Signals
############################################################

accum_act_raddr = WireVector(ACCUMSIZE)  # Activate unit read address for accumulator buffers
weights_in = Input(MATSIZE*DWIDTH, "weights_in")

############################################################
#  Instruction Memory and PC
############################################################

IMem = MemBlock(bitwidth=INSTRUCTION_WIDTH, addrwidth=IMEM_SIZE)
pc = Register(IMEM_SIZE)
pc.incr = WireVector(1)
with conditional_assignment:
    with pc.incr:
        pc.next |= pc + 1

############################################################
#  Unified Buffer
############################################################

UBuffer = MemBlock(bitwidth=MATSIZE*DWIDTH, addrwidth=UBSIZE)

# Address and data wires for MM read port
ub_mm_raddr = WireVector(UBuffer.addrwidth)  # MM UB read address
UB2MM = UBuffer[ub_mm_raddr]

############################################################
#  Decoder
############################################################

dispatch_mm, dispatch_act, ub_start_addr, ub_dec_raddr, ub_dest_addr, rhm_length, whm_length, mmc_length, act_length, accum_raddr, accum_waddr, accum_overwrite, switch_weights, weights_we = decode(IMem[pc])

############################################################
#  Matrix Multiply Unit
############################################################

ub_mm_raddr_sig, acc_out, mm_busy, mm_done = MMU_top(data_width=DWIDTH, matrix_size=MATSIZE, accum_size=ACCUMSIZE, ub_size=UBSIZE, start=dispatch_mm, start_addr=ub_start_addr, nvecs=mmc_length, dest_acc_addr=accum_waddr, overwrite=accum_overwrite, swap_weights=switch_weights, ub_rdata=UB2MM, accum_raddr=accum_act_raddr, weights_in=weights_in, weights_we=weights_we)

ub_mm_raddr <<= ub_mm_raddr_sig

############################################################
#  Activate Unit
############################################################

accum_raddr_sig, ub_act_waddr, act_out, ub_act_we, act_busy = act_top2(start=dispatch_act, start_addr=accum_raddr, dest_addr=ub_dest_addr, nvecs=act_length, accum_out=acc_out)
accum_act_raddr <<= accum_raddr_sig

# Write the result of activate to the unified buffer
with conditional_assignment:
    with ub_act_we:
        UBuffer[ub_act_waddr] |= act_out
