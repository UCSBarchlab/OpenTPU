# Function: Relu and normalization. 
# It reads from Accum Buffer and writes to the Unified Buffer. 
# Comments: offset defined during design phase (not runtime). 
# Code for "dynamic" normalization available as well. 
# Simulation seems to work well. Reordering may be REQUIRED though.

import pyrtl

# relu and normalization
def relu_elem(din, offset):
	assert offset <= 24
	dout = pyrtl.WireVector(32)
	dout_reg = pyrtl.Register(8)
	with pyrtl.conditional_assignment: 
		with din[-1] == 0: 
			dout |= din
		with pyrtl.otherwise:
			dout |= 0 
	dout_reg.next <<= dout[24-offset:32-offset]
	return dout_reg

# Latency: 1cc
def relu_vector(din, offset=0):
	dout = [relu_elem(din[i], offset) for i in range(len(din))]
	return dout

# Latency: N+1 cc
def act_top(rd_addr, N, wr_addr):
	cntr = pyrtl.Register(2, name='counter')
	cntr_dl1 = pyrtl.Register(2)
	cntr_dl2 = pyrtl.Register(2)
	din_addr = pyrtl.Register(2, name='din_addr')
	dout_addr = pyrtl.Register(2, name='dout_addr')
	dout = pyrtl.Register(4*8, name='dout')
        din = []
        din_addr.next |= rd_addr + cntr
	with pyrtl.conditional_assignment:
		with cntr < N-1:
			cntr.next |= cntr + 1
		with pyrtl.otherwise:
			cntr.next |= 0
	for i in range(4):
		din.append(mem_acum[din_addr][i*32:(i+1)*32-1])
	# latency from read to write: 2cc
	cntr_dl1.next <<= cntr
	cntr_dl2.next <<=cntr_dl1
	dout_addr.next <<= wr_addr + cntr_dl2
	relu_out = relu_vector(din, 24)
	dout.next <<= pyrtl.concat_list(relu_out)
	#mem_ub[dout_addr] <<= pyrtl.concat_list(relu_out)
	return 1	
	
# Test
mem_acum = pyrtl.MemBlock(bitwidth=4*32,addrwidth=2)
mem_acum[0] <<= pyrtl.Const(0x00000001000000020000000300000004)
mem_acum[1] <<= pyrtl.Const(0x00000005000000060000000700000008)
mem_acum[2] <<= pyrtl.Const(0x000000090000000A0000000B0000000C)
mem_acum[3] <<= pyrtl.Const(0x0000000D0000000E0000000F00000010)
mem_ub = pyrtl.MemBlock(bitwidth=4*8,addrwidth=2)

act_top(0,4,0)

# simulate the instantiated design for 15 cycles
sim_trace = pyrtl.SimulationTrace()
sim = pyrtl.Simulation(tracer=sim_trace)
for cyle in range(15):
	sim.step({})
sim_trace.render_trace()  
