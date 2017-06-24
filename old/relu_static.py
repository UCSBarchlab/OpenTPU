# Function: Relu and normalization
# Comments: offset defined during design phase (not runtime)

import pyrtl

# relu and normalization
def relu_nrml(din, offset=0):
 	assert len(din) == 32 
	assert offset <= 24
	dout = pyrtl.WireVector(32)
	with pyrtl.conditional_assignment: 
		with din[-1] == 0: 
			dout |= din
		with pyrtl.otherwise:
			dout |= 0 
	return dout[24-offset:32-offset]

# Test: collects only the 8 LSBs (after relu)
relu_in = pyrtl.Register(bitwidth=32, name='din')
relu_in.next <<= 300
offset = 24
dout = relu_nrml(relu_in, offset)
relu_out = pyrtl.Register(bitwidth=8, name='dout')
relu_out.next <<= dout 

# simulate the instantiated design for 15 cycles
sim_trace = pyrtl.SimulationTrace()
sim = pyrtl.Simulation(tracer=sim_trace)
for cyle in range(35):
	sim.step({})
sim_trace.render_trace()  
