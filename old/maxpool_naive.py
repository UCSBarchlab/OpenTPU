# We're not sorry for the TERRIBLE code! 
# We now know how to create lists containing Registers/WireVectors :)
# The pipeline was also implemented traditionally.
# Attention: parametric code!
# Function: maxpooling
# Design: fully-pipelined. Latency: ceil(log(n))
# v2 it is, v3 is coming and it looks much better! We promise :)

import pyrtl

def bitCmp(din0, din1):
	dout = pyrtl.WireVector(32)
        dout_reg = pyrtl.Register(32)
	with pyrtl.conditional_assignment: 
		with din0 >= din1: 
			dout |= din0
		with pyrtl.otherwise:
			dout |= din1
	dout_reg.next <<= dout
	return dout_reg

def maxpool(din):
	if (len(din)==1): 
		return din[0]
	elif (len(din)==2):
		return bitCmp(din[0], din[1])
	else:
		left = maxpool(din[:len(din)/2])
		right = maxpool(din[len(din)/2:])
		return bitCmp(left, right)

	

# instantiate relu and set test inputs
'''din = []
din0 = pyrtl.Register(bitwidth=32, name='din0')
din0.next <<= 10
din1 = pyrtl.Register(bitwidth=32, name='din1')
din1.next <<= 12
din2 = pyrtl.Register(bitwidth=32, name='din2')
din2.next <<= 10
din3 = pyrtl.Register(bitwidth=32, name='din3')
din3.next <<= 12
din4 = pyrtl.Register(bitwidth=32, name='din4')
din4.next <<= 127
din5 = pyrtl.Register(bitwidth=32, name='din5')
din5.next <<= 12
din6 = pyrtl.Register(bitwidth=32, name='din6')
din6.next <<= 10
din7 = pyrtl.Register(bitwidth=32, name='din7')
din7.next <<= 12
din8 = pyrtl.Register(bitwidth=32, name='din8')
din8.next <<= 10


din.append(din0)
din.append(din1)
din.append(din2)
din.append(din3)
din.append(din4)
din.append(din5)
din.append(din6)
din.append(din7)
din.append(din8)


#for i in range(9):
	#din.append(pyrtl.Register(bitwidth=32, name='dins'))


dout = maxpool(din)
cmpr_out = pyrtl.Register(bitwidth=32, name='cmpr_out')
cmpr_out.next <<= dout 

# simulate the instantiated design for 15 cycles
sim_trace = pyrtl.SimulationTrace()
sim = pyrtl.Simulation(tracer=sim_trace)
for cyle in range(35):
	sim.step({})
sim_trace.render_trace() '''
