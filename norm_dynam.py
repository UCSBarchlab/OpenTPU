'''
Function: Normalization
Design: fully-pipelined (it may be better to replace some Regs with WireVectors
Comments: the offset can change at runtime. To avoid the use of big MUXes we have used a barrel shifter. PyRTL rtllib contains a barrel_shifter, but the interpreter was returing us an error when trying to call it from the lib, so we copied the whole function here.
'''

import pyrtl
import math

def barrel_shifter(shift_in, bit_in, direction, shift_dist, wrap_around=0):
    """
    Create a barrel shifter that operates on data based on the wire width
    :param shift_in: the input wire
    :param bit_in: the 1-bit wire giving the value to shift in
    :param direction: a one bit WireVector representing shift direction
        (0 = shift down, 1 = shift up)
    :param shift_dist: WireVector representing offset to shift
    :param wrap_around: ****currently not implemented****
    :return: shifted WireVector
    """
    # Implement with logN stages pyrtl.muxing between shifted and un-shifted values

    val = shift_in
    append_val = bit_in
    log_length = int(math.log(len(shift_in)-1, 2))  # note the one offset

    if len(shift_dist) > log_length:
        print('Warning: for barrel shifter, the shift distance wirevector '
              'has bits that are not used in the barrel shifter')

    for i in range(min(len(shift_dist), log_length)):
        shift_amt = pow(2, i)  # stages shift 1,2,4,8,...
        newval = pyrtl.select(direction, truecase=val[:-shift_amt], falsecase=val[shift_amt:])
        newval = pyrtl.select(direction, truecase=pyrtl.concat(newval, append_val),
                              falsecase=pyrtl.concat(append_val, newval))  # Build shifted value
        # pyrtl.mux shifted vs. unshifted by using i-th bit of shift amount signal
        val = pyrtl.select(shift_dist[i], truecase=newval, falsecase=val)
        append_val = pyrtl.concat(append_val, bit_in)

    return val

# main normalization module
def nrml(din, offset=24):
    zero = pyrtl.Const(0,1)
    one = pyrtl.Const(1,1)
    temp = pyrtl.Register(32, name='temp')
    temp.next <<= barrel_shifter(din, zero, zero, offset)
    dout = pyrtl.Register(8, name='dout')
    dout.next <<= temp[:8]
    return dout

# test
din = pyrtl.Register(bitwidth=32, name='din')
din.next <<= 300
offset = pyrtl. Register(bitwidth = 32, name= 'offset')
offset.next <<= 5 
test_out = nrml(din,offset)

# simulate the instantiated design for 15 cycles
sim_trace = pyrtl.SimulationTrace()
sim = pyrtl.Simulation(tracer=sim_trace)
for cyle in range(35):
	sim.step({})
sim_trace.render_trace() 

