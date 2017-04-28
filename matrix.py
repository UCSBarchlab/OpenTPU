from pyrtl import *

#set_debug_mode()

def MAC(data_in, acc_in, switchw, weight_in, weight_we, weight_tag):
    '''Multiply-Accumulate unit with programmable weight.
    Inputs
    data_in: The 8-bit activation value to multiply by weight.
    acc_in: 32-bit value to accumulate with product.
    switchw: Control signal; when 1, switch to using the other weight buffer.
    weight_in: 8-bit value to write to the secondary weight buffer.
    weight_we: When high, weights are being written; if tag matches, store weights.
               Otherwise, pass them through with incremented tag.
    weight_tag: If equal to 255, weight is for this row; store it.

    Outputs
    out: Result of the multiply accumulate; moves one cell down to become acc_in.
    data_reg: data_in, stored in a pipeline register for cell to the right.
    switch_reg: switchw, stored in a pipeline register for cell to the right.
    weight_reg: weight_in, stored in a pipeline register for cell below.
    weight_we_reg: weight_we, stored in a pipeline register for cell below.
    weight_tag_reg: weight_tag, incremented and stored in a pipeline register for cell below
    '''

    # Check lengths of inupts
    if len(weight_in) != len(data_in) != 8:
        raise Exception("Expected 8-bit value in MAC.")
    if len(switchw) != len(weight_we) != 1:
        raise Exception("Expected 1-bit control signal in MAC.")

    # Should never switch weight buffers while they're changing
    rtl_assert(weight_we & switchw, Exception("Cannot switch weight values when they're being loaded!"))

    # Use two buffers to store weight and next weight to use.
    wbuf1, wbuf2 = Register(len(weight_in)), Register(len(weight_in))

    # Track which buffer is current and which is secondary.
    current_buffer_reg = Register(1)
    with conditional_assignment:
        with switchw:
            current_buffer_reg.next |= ~current_buffer_reg
    current_buffer = current_buffer_reg ^ switchw  # reflects change in same cycle switchw goes high

    # When told, store a new weight value in the secondary buffer
    with conditional_assignment:
        with weight_we & (weight_tag == Const(255)):
            with current_buffer == 0:  # If 0, wbuf1 is current; if 1, wbuf2 is current
                wbuf2.next |= weight_in
            with otherwise:
                wbuf1.next |= weight_in

    # For values that need to be forward to the right/bottom, store in pipeline registers
    data_reg = Register(len(data_in))  # pipeline register, holds data value for cell to the right
    data_reg.next <<= data_in
    switch_reg = Register(1)  # pipeline register, holds switch control signal for cell to the right
    switch_reg.next <<= switchw
    weight_reg = Register(len(weight_in))  # pipeline register, holds weight input for cell below
    weight_reg.next <<= weight_in
    weight_we_reg = Register(1)  # pipeline register, holds weight write enable signal for cell below
    weight_we_reg.next <<= weight_we
    weight_tag_reg = Register(len(weight_tag))  # pipeline register, holds weight tag for cell below
    weight_tag_reg.next <<= (weight_tag + 1)[:len(weight_tag)]  # increment tag as it passes down rows

    # Do the actual MAC operation
    weight = select(current_buffer, wbuf2, wbuf1)
    product = weight * data_in
    out = product + acc_in
    if len(out) > 32:
        out = out[:32]

    return out, data_reg, switch_reg, weight_reg, weight_we_reg, weight_tag_reg


def MMArray(data_in, new_weights, weights_in, weights_we):
    '''
    data_in: 256-array of 8-bit activation values from systolic_setup buffer
    new_weights: 256-array of 1-bit control values indicating that new weight should be used
    weights_in: 65,536-array of 8-bit weights (output of Weight FIFO)
    weights_we: 1-bit signal to begin writing new weights into the matrix
    '''

    # For signals going to the right, store in a var; for signals going down, keep a list
    # For signals going down, keep a copy of inputs to top row to connect to later
    weights_in_top = [ WireVector(8) for i in range(256) ]  # input weights to top row
    weights_in_last = [x for x in weights_in_top]
    weights_enable_top = [ WireVector(1) for i in range(256) ]  # weight we to top row
    weights_enable = [x for x in weights_enable_top]
    weights_tag_top = [ WireVector(8) for i in range(256) ]  # weight row tag to top row
    weights_tag = [x for x in weights_tag_top]
    data_out = [None for i in range(256)]  # will hold output from final row
    # Build array of MACs
    for i in range(256):  # for each row
        din = data_in[i]
        acc_in = Const(0)
        switchin = new_weights[i]
        for j in range(256):  # for each column
            acc_in, din, switchin, newweight, newwe, newtag  = MAC(din, acc_in, switchin, weights_in_last[j], weights_enable[j], weights_tag[j])
            weights_in_last[j] = newweight
            weights_enable[j] = newwe
            weights_tag[j] = newtag
            data_out[j] = acc_in
    
    # Handle weight reprogramming
    programming = Register(1)  # when 1, we're in the process of loading new weights
    progstep = Register(8)  # 256 steps to program new weights (also serves as tag input)
    with conditional_assignment:
        with weights_we & (~programming):
            programming.next |= 1
        with programming & (progstep == 255):
            programming.next |= 0
        with otherwise:
            pass
        with programming:  # while programming, increment state each cycle
            progstep.next |= progstep + 1

    # Divide FIFO out into 256 256-entry arrays
    # Combine rows into single 2048-bit wires
    weight_arr = [ concat_list([weights_in[j*256+i] for i in range(256)]) for j in range(256) ]
    # Mux the 2048-bit wire for this row
    current_weights_wire = mux(progstep, *weight_arr)
    # Split the wire into an array of 8-bit values
    current_weights = [ current_weights_wire[i*8:i*8+8] for i in range(256) ]

    # Connect top row to input and control signals
    for i, win in enumerate(weights_in_top):
        # From the current 256-array, select the byte for this column
        win <<= current_weights[i]
    for we in weights_enable_top:
        # Whole row gets same signal: high when programming new weights
        we <<= programming
    for wt in weights_tag_top:
        # Tag is same for whole row; use state index (runs from 0 to 255)
        wt <<= progstep

    return data_out


def systolic_setup():
    '''Buffers vectors from the unified SRAM buffer so that they can be fed along diagonals to the
    Matrix Multiply array.
    '''
    
    return

def FIFO():
    return

def accumulators():
    return


'''
Control signals propagating down systolic_setup to accumulators:
-Overwrite signal (default: accumulate)
-New accumulator address value (default: add 1 to previous address)
-Done signal?
'''

ins = [Input(8) for i in range(256)]
swap = Input(1, 'swap')
ws = [Const(i, bitwidth=8) for i in range(256)] * 256
we = Input(1, 'we')

outs = [Output(32) for i in range(256)]
mouts = MMArray(ins, [swap,]*256, ws, we)
for x,y in zip(outs, mouts):
    x <<= y

sim_trace = SimulationTrace()
sim = FastSimulation(tracer=sim_trace)

# First, send signal to write weights
d = {ins[j] : 0 for j in range(256) }
d.update({ we : 1, swap : 0 })
sim.step(d)

# Wait 256 cycles for weights to propagate
for i in range(260):
    d = {ins[j] : 0 for j in range(256) }
    d.update({ we : 0, swap : 0 })
    sim.step(d)

# Send the swap signal with first row of input
d = {ins[j] : j for j in range(256) }
d.update({ we : 0, swap : 1 })
sim.step(d)

# Send 255 more rows of input
for i in range(255):
    d = {ins[j] : j for j in range(256) }
    d.update({ we : 0, swap : 0 })
    sim.step(d)

# Wait some cycles while it propagates
for i in range(300):
    d = {ins[j] : 0 for j in range(256) }
    d.update({ we : 0, swap : 0 })
    sim.step(d)

with open('trace.vcd', 'w') as f:
    sim_trace.print_vcd(f)
