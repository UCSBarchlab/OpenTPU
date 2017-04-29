# Sorry for the TERRIBLE code! 
# I still don't know how to create lists containing Registers/WireVectors :)
# The pipeline was also implemented traditionally.
# Attention: Non-parametric code!
# Function: maxpooling
# Design: fully-pipelined. Latency: ceil(log(n)), where n = 9 (fixed)
# v2 is coming and it looks much better! We promise :)

import pyrtl

def bitCmp(din0, din1):
	dout = pyrtl.WireVector(32)
	with pyrtl.conditional_assignment: 
		with din0 >= din1: 
			dout |= din0
		with pyrtl.otherwise:
			dout |= din1
	return dout

def maxpool(din0, din1, din2, din3, din4, din5, din6, din7, din8):
	# 1st stage
	temp00 = bitCmp(din0, din1)
	temp01 = bitCmp(din2, din3)
	temp02 = bitCmp(din4, din5)
	temp03 = bitCmp(din6, din7)
	temp04 = din8
	temp00_reg = pyrtl.Register(bitwidth=32, name = 'temp00')
	temp00_reg.next <<= temp00
	temp01_reg = pyrtl.Register(bitwidth=32, name = 'temp01')
	temp01_reg.next <<= temp01
	temp02_reg = pyrtl.Register(bitwidth=32, name = 'temp02')
	temp02_reg.next <<= temp02
	temp03_reg = pyrtl.Register(bitwidth=32, name = 'temp03')
	temp03_reg.next <<= temp03
	temp04_reg = pyrtl.Register(bitwidth=32, name = 'temp04')
	temp04_reg.next <<= temp04
	# 2nd stage
	temp10 = bitCmp(temp00_reg, temp01_reg)
	temp11 = bitCmp(temp02_reg, temp03_reg)
	temp12 = temp04_reg
	temp10_reg = pyrtl.Register(bitwidth=32, name = 'temp10')
	temp10_reg.next <<= temp10
	temp11_reg = pyrtl.Register(bitwidth=32, name = 'temp11')
	temp11_reg.next <<= temp11
	temp12_reg = pyrtl.Register(bitwidth=32, name = 'temp12')
	temp12_reg.next <<= temp12
	# 3rd stage
	temp20 = bitCmp(temp10_reg, temp11_reg)
	temp21 = temp12_reg
	temp20_reg = pyrtl.Register(bitwidth=32)
	temp20_reg.next <<= temp20
	temp21_reg = pyrtl.Register(bitwidth=32)
	temp21_reg.next <<= temp21
	# 4th stage
	dout = bitCmp(temp20_reg, temp21_reg)
	return dout	


# instantiate relu and set test inputs
din0 = pyrtl.Register(bitwidth=32, name='din0')
din1 = pyrtl.Register(bitwidth=32, name='din1')
din2 = pyrtl.Register(bitwidth=32, name='din2')
din3 = pyrtl.Register(bitwidth=32, name='din3')
din4 = pyrtl.Register(bitwidth=32, name='din4')
din5 = pyrtl.Register(bitwidth=32, name='din5')
din6 = pyrtl.Register(bitwidth=32, name='din6')
din7 = pyrtl.Register(bitwidth=32, name='din7')
din8 = pyrtl.Register(bitwidth=32, name='din8')
din0.next <<= 10
din1.next <<= 12
din2.next <<= 5
din3.next <<= 8
din4.next <<= 11
din5.next <<= 14
din6.next <<= 3
din7.next <<= 6
din8.next <<= 9
dout = maxpool(din0, din1, din2, din3, din4, din5, din6, din7, din8)
cmpr_out = pyrtl.Register(bitwidth=32, name='cmpr_out')
cmpr_out.next <<= dout 

# simulate the instantiated design for 15 cycles
sim_trace = pyrtl.SimulationTrace()
sim = pyrtl.Simulation(tracer=sim_trace)
for cyle in range(35):
	sim.step({})
sim_trace.render_trace() 
