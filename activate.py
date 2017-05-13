
#import pyrtl
from pyrtl import *

def relu_vector(vec, offset):
    assert offset <= 24
    return [ select(d[-1], falsecase=d, truecase=Const(0, len(d)))[24-offset:32-offset] for d in vec ]
    
def act_top(start, start_addr, dest_addr, nvecs, accum_out):

    busy = Register(1)
    accum_addr = Register(len(start_addr))
    ub_waddr = Register(len(dest_addr))
    N = Register(len(nvecs))
    
    rtl_assert(~(start & busy), Exception("Dispatching new activate instruction while previous instruction is still running."))
    
    with conditional_assignment:
        with start:  # new instruction being dispatched
            accum_addr.next |= start_addr
            ub_waddr.next |= dest_addr
            N.next |= nvecs
            busy.next |= 1
        with busy:  # Do activate on another vector this cycle
            accum_addr.next |= accum_addr + 1
            ub_waddr.next |= ub_waddr + 1
            N.next |= N - 1
            with N == 1:  # this was the last vector
                busy.next |= 0

    act_out_list = relu_vector(accum_out, 24)
    act_out = concat_list(act_out_list)
    ub_we = busy
            
    return accum_addr, ub_waddr, act_out, ub_we, busy
