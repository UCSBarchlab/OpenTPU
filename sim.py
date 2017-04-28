# GLOBALS:
weight_memory = np.load('dram.npy')
host_memory = np.load('hostmem.npy')
unified_buffer = np.zeros((96000, 256))
accumulator = np.zeros((4000, 256))
weight_fifo = []

# read text file with program
with open(sys.argv[1]) as f:
	for line in f:
		v = line.split(" ")
		opcode = v[0]
		operands = v[1:]

# opcodes
def read_host_mem(host_start_addr, ub_start_addr, size):
	'''
	Reads a range of memory addresses from "start_addr" to "stop_addr" in the
	host_memory
	'''
	unified_buffer[ub_start_addr:ub_start_addr+size] = host_memory[host_start:host_start+size]
	print("Host Memory Read Complete")

def write_host_mem(ub_start_addr, host_start, size):
	'''
	Writes "data" to a range of memory addresses from "start_addr" to
	"stop_addr" in host_memory
	'''
	host_memory[host_start:host_start+size] = unified_buffer[ub_start_addr:ub_start_addr+size]
	print("Host Memory Write Complete")

def read_weights(weight_dram_addr):
	'''
	Moves one tiles into the weight fifo
	'''
	weight_fifo.append(weight_memory[weight_dram_addr])

def matrix_multiply_convolve(switch, ub_start_addr, accumulator_addr, size, weight_fifo):
	inp = unified_buffer[ub_start_addr: ub_start_addr+size]
	out = np.matmul(inp, weight_fifo.pop())
	accumulator[accumulator_addr:accumulator_addr+256] = out# because matmul unit is 256*256
