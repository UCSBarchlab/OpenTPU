from pyrtl import *

from isa import INSTRUCTION_WIDTH
from decoder import decode
from matrix import MMU_top

############################################################
#  Machine Parameters
############################################################

DWIDTH = 8  # 8
MATSIZE = 4  # 256
ACCUMSIZE = 8  # 13 (4k entries)
UBSIZE = 15  #  (96k entries in original; need one 2^16 and one 2^15 memory for that)
IMEM_SIZE = 15

############################################################
#  Control Signals
############################################################

dispatch = WireVector(1)  # raise high when dispatching instruction
accum_act_raddr = WireVector(ACCUMSIZE)  # Activate unit read address for accumulator buffers
weights_in = Input(MATSIZE*DATWIDTH, "weights_in")

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

ub_start_addr, ub_dec_raddr, ub_dest_addr, rhm_length, whm_length, mmc_length, act_length, accum_waddr, accum_overwrite, switch_weights, weights_we = decode(IMem[pc])

############################################################
#  Matrix Multiply Unit
############################################################

ub_mm_raddr_sig, acc_out, mm_busy, mm_done = MMU_top(data_width=DWIDTH, matrix_size=MATSIZE, accum_size=ACCUMSIZE, ub_size=UBSIZE, init=dispatch, start_addr=ub_start_addr, nvecs=mmc_length, dest_acc_addr=accum_waddr, overwrite=accum_overwrite, swap_weights=switch_weights, ub_rdata=UB2MM, accum_raddr=accum_act_raddr, weights_in=weights_in, weights_we=weights_we)

ub_mm_raddr <<= ub_mm_raddr_sig
