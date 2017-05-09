from pyrtl import *

from config import *
from decoder import decode
from matrix import MMU_top
from act_top import act_top2

############################################################
#  Control Signals
############################################################

accum_act_raddr = WireVector(ACC_ADDR_SIZE)  # Activate unit read address for accumulator buffers
weights_in = Input(MATSIZE*DWIDTH, "weights_in")

############################################################
#  Instruction Memory and PC
############################################################

IMem = MemBlock(bitwidth=INSTRUCTION_WIDTH, addrwidth=IMEM_ADDR_SIZE)
pc = Register(IMEM_ADDR_SIZE)
pc.incr = WireVector(1)
with conditional_assignment:
    with pc.incr:
        pc.next |= pc + 1

############################################################
#  Unified Buffer
############################################################

UBuffer = MemBlock(bitwidth=MATSIZE*DWIDTH, addrwidth=UB_ADDR_SIZE, max_write_ports=2)

# Address and data wires for MM read port
ub_mm_raddr = WireVector(UBuffer.addrwidth)  # MM UB read address
UB2MM = UBuffer[ub_mm_raddr]

############################################################
#  Decoder
############################################################

dispatch_mm, dispatch_act, dispatch_rhm, dispatch_whm, ub_start_addr, ub_dec_addr, ub_dest_addr, rhm_dec_addr, whm_dec_addr, rhm_length, whm_length, mmc_length, act_length, accum_raddr, accum_waddr, accum_overwrite, switch_weights, weights_we = decode(IMem[pc])

############################################################
#  Matrix Multiply Unit
############################################################

ub_mm_raddr_sig, acc_out, mm_busy, mm_done = MMU_top(data_width=DWIDTH, matrix_size=MATSIZE, accum_size=ACC_ADDR_SIZE, ub_size=UB_ADDR_SIZE, start=dispatch_mm, start_addr=ub_start_addr, nvecs=mmc_length, dest_acc_addr=accum_waddr, overwrite=accum_overwrite, swap_weights=switch_weights, ub_rdata=UB2MM, accum_raddr=accum_act_raddr, weights_in=weights_in, weights_we=weights_we)

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

############################################################
#  Read/Write Host Memory
############################################################

hostmem_raddr = Output(HOST_ADDR_SIZE)
hostmem_rdata = Input(DWIDTH*MATSIZE)
hostmem_re = Output(1)
hostmem_waddr = Output(HOST_ADDR_SIZE)
hostmem_wdata = Output(DWIDTH*MATSIZE)
hostmem_we = Output(1)

# Write Host Memory control logic
whm_N = Register(len(whm_length))
whm_ub_raddr = Register(len(ub_dec_addr))
whm_addr = Register(len(whm_dec_addr))
whm_busy = Register(1)

ubuffer_out = UBuffer[whm_ub_raddr]

hostmem_waddr <<= whm_addr
hostmem_wdata <<= ubuffer_out

with conditional_assignment:
    with dispatch_whm:
        whm_N.next |= whm_length
        whm_ub_raddr.next |= ub_dec_addr
        whm_addr.next |= whm_dec_addr
        whm_busy.next |= 1
    with whm_busy:
        whm_N.next |= whm_N - 1
        whm_ub_raddr.next |= whm_ub_raddr + 1
        whm_addr.next |= whm_addr + 1
        hostmem_we |= 1
        with whm_N == 1:
            whm_busy.next |= 0


# Read Host Memory control logic
rhm_N = Register(len(rhm_length))
rhm_addr = Register(len(rhm_dec_addr))
rhm_busy = Register(1)
with conditional_assignment:
    with dispatch_rhm:
        rhm_N.next |= rhm_length
        rhm_busy.next |= 1
        hostmem_raddr |= rhm_dec_addr
        hostmem_re |= 1
        rhm_addr.next |=  + 1
    with rhm_busy:
        rhm_N.next |= rhm_N - 1
        hostmem_raddr |= rhm_addr
        hostmem_re |= 1
        rhm_addr.next |= rhm_addr + 1
        UBuffer[ub_dec_addr] |= hostmem_rdata
        with rhm_N == 1:
            rhm_busy.next |= 0
