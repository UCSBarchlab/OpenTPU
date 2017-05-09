from pyrtl import *
from tpu import *
from config import INSTRUCTION_WIDTH

import sys


with open(sys.argv[1], 'rb') as f:
    ins = map(ord, f.read())  # create byte list from input

instrs = []
width = INSTRUCTION_WIDTH / 8
# This assumes instructions are strictly byte-aligned
for i in range(len(ins)/width):  # once per instruction
    val = 0
    for j in range(width):  # for each byte
        val = (val << 8) | ins.pop(0)
    instrs.append(val)

print map(hex, instrs)

sim_trace = SimulationTrace({IMem : { a : v for a,v in enumerate(instrs)} })
sim = FastSimulation(tracer=sim_trace)


