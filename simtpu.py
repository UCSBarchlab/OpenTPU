from pyrtl import *

#set_debug_mode()

from tpu import *
from config import INSTRUCTION_WIDTH

import sys



with open(sys.argv[1], 'rb') as f:
    ins = [x for x in f.read()]  # create byte list from input

instrs = []
width = INSTRUCTION_WIDTH / 8
# This assumes instructions are strictly byte-aligned


for i in range(int(len(ins)/width)):  # once per instruction
    val = 0
    for j in range(int(width)):  # for each byte
        val = (val << 8) | ins.pop(0)
    instrs.append(val)

print(list(map(hex, instrs)))
    
sim_trace = SimulationTrace()
sim = FastSimulation(tracer=sim_trace, memory_value_map={ IMem : { a : v for a,v in enumerate(instrs)} })

d = {
    weights_in : 0,
    hostmem_rdata : 0,
}
for i in range(100):
    sim.step(d)

#sim_trace.render_trace()
with open("trace.vcd", 'w') as f:
    sim_trace.print_vcd(f)
