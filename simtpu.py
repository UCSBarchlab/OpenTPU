from pyrtl import *
import argparse
import numpy as np

#set_debug_mode()

from tpu import *
from config import INSTRUCTION_WIDTH

import sys

parser = argparse.ArgumentParser(description="Run the PyRTL spec for the TPU on the indicated program.")
parser.add_argument("prog", metavar="program.bin", help="A valid binary program for OpenTPU.")
parser.add_argument("hostmem", metavar="HostMemoryArray", help="A file containing a numpy array containing the initial contents of host memory. Each row represents one vector.")
parser.add_argument("weightsmem", metavar="WeightsMemoryArray", help="A file containing a numpy array containing the contents of the weights memroy. Each row represents one tile (the first row corresponds to the top row of the weights matrix).")

args = parser.parse_args()

# Read the program and build an instruction list
with open(args.prog, 'rb') as f:
    ins = [x for x in f.read()]  # create byte list from input

instrs = []
width = INSTRUCTION_WIDTH / 8
# This assumes instructions are strictly byte-aligned

for i in range(int(len(ins)/width)):  # once per instruction
    val = 0
    for j in range(int(width)):  # for each byte
        val = (val << 8) | ins.pop(0)
    instrs.append(val)

#print(list(map(hex, instrs)))

# Read the dram files and build memory images
#with open(args.hostmem, 'r') as f:
#    hostarray = np.load(f)
hostarray = np.load(args.hostmem)
#print(hostarray)
print(hostarray.shape)

weightsarray = np.load(args.weightsmem)
#print(weightsarray)
print(weightsarray.shape)

# Run Simulation
sim_trace = SimulationTrace()
sim = FastSimulation(tracer=sim_trace, memory_value_map={ IMem : { a : v for a,v in enumerate(instrs)} })

d = {
    weights_in : 0,
    hostmem_rdata : 0,
}

sim.step(d)

while True:
    if sim.inspect(halt):
        break

    sim.step(d)


#sim_trace.render_trace()
with open("trace.vcd", 'w') as f:
    sim_trace.print_vcd(f)
