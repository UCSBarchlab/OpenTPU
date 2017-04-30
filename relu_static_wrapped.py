# Function: Relu and normalization. Start and done signals included
# Latency: 1cc
# Comments: offset defined during design phase (not runtime)

import pyrtl

# relu and normalization
def relu_nrml(din, offset):
 	assert len(din) == 32 
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

def relu_top(din, start, offset=0):
	dout = [relu_nrml(din[i], offset) for i in range(len(din))]
	done = pyrtl.Register(1)
	done.next <<= start
	return done, dout

# Test: collects only the 8 LSBs (after relu)
relu_in = []
relu_in0 = pyrtl.Register(bitwidth=32, name='din0')
relu_in0.next <<= 300
relu_in.append(relu_in0)
relu_in1 = pyrtl.Register(bitwidth=32, name='din1')
relu_in1.next <<= 200
relu_in.append(relu_in1)
relu_in2 = pyrtl.Register(bitwidth=32, name='din2')
relu_in2.next <<= 100
relu_in.append(relu_in2)
start = pyrtl.Register(bitwidth=1, name='start')
start.next <<= 1
offset = 24
done, dout = relu_top(relu_in, start, offset)
relu_out0 = pyrtl.Register(bitwidth=8, name='dout0')
relu_out0.next <<= dout[0]
relu_out1 = pyrtl.Register(bitwidth=8, name='dout1')
relu_out1.next <<= dout[1] 
relu_out2 = pyrtl.Register(bitwidth=8, name='dout2')
relu_out2.next <<= dout[2]
relu_done = pyrtl.Register(bitwidth=1, name='done')
relu_done.next <<= done

# simulate the instantiated design for 15 cycles
sim_trace = pyrtl.SimulationTrace()
sim = pyrtl.Simulation(tracer=sim_trace)
for cyle in range(35):
	sim.step({})
sim_trace.render_trace()  
