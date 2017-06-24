
#import pyrtl
from pyrtl import *

def relu_vector(vec, offset):
    assert offset <= 24
    return concat_list([ select(d[-1], falsecase=d, truecase=Const(0, len(d)))[24-offset:32-offset] for d in vec ])

def sigmoid(x):
    rb = RomBlock(bitwidth=8, addrwidth=3, asynchronous=True, romdata={0: 128, 1: 187, 2: 225, 3: 243, 4: 251, 5: 254, 6: 255, 7: 255, 8: 255})
    x_gt_7 = reduce(lambda x, y: x|y, x[3:])  # OR of bits 3 and up
    return select(x_gt_7, falsecase=rb[x[:3]], truecase=Const(255, bitwidth=8))

def sigmoid_vector(vec):
    return concat_list([ sigmoid(x) for x in vec ])


def act_top(start, start_addr, dest_addr, nvecs, func, accum_out):

    # func: 0 - nothing
    #       1 - ReLU
    #       2 - sigmoid

    busy = Register(1)
    accum_addr = Register(len(start_addr))
    ub_waddr = Register(len(dest_addr))
    N = Register(len(nvecs))
    act_func = Register(len(func))
    
    rtl_assert(~(start & busy), Exception("Dispatching new activate instruction while previous instruction is still running."))
    
    with conditional_assignment:
        with start:  # new instruction being dispatched
            accum_addr.next |= start_addr
            ub_waddr.next |= dest_addr
            N.next |= nvecs
            act_func.next |= func
            busy.next |= 1
        with busy:  # Do activate on another vector this cycle
            accum_addr.next |= accum_addr + 1
            ub_waddr.next |= ub_waddr + 1
            N.next |= N - 1
            with N == 1:  # this was the last vector
                busy.next |= 0

    invals = concat_list([ x[:8] for x in accum_out ])
    act_out = mux(act_func, invals, relu_vector(accum_out, 24), sigmoid_vector(accum_out), invals)
    #act_out = relu_vector(accum_out, 24)
    ub_we = busy
            
    return accum_addr, ub_waddr, act_out, ub_we, busy
