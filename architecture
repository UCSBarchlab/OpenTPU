
### Writing a Program
OpenTPU uses no dynamic scheduling; all execution is fully determinstic* and the hardware relies on the compiler to correctly schedule operations and pad NOPs to handle delays. This OpenTPU release does not support "repeat" flags on instructions, so many NOPs are required to ensure correct execution.

*DRAM is a source of non-deterministic latency, discussed in the Memory Controller section of Microarchitecture.


### Latencies
The following gives the hardware execution latency for each instruction on OpenTPU:

RHM - _M_ cycles for reading _M_ vectors
WHM - _M_ cycles for writing _M_ vectors
RW - _N*N_/64 cycles for _N_x_N_ MM Array for DRAM transfer, and up to 3 additional cycles to propagate through the FIFO
MMC - _L+2N_ cycles, for _N_x_N_ MM Array and _L_ vectors multiplied in the instruction
ACT - _L+1_ cycles, for _L_ vectors activated in the instruction


## Microarchitecture

### Matrix Multiply (MM) Unit
The core of the compute of the OpenTPU is the parametrizable array of 8-bit Multiply-Accumulate Units (MACs), each consisting of an 8-bit integer multiplier and an integer adder of between 16 and 32 bits*. Each MAC has two buffers storing 8-bit weights (the second buffer allows weight programming to happen in parallel). Input vectors enter the array from the left, with values advancing one unit to the right each cycle. Each unit multiplies the input value by the active weight, adds it to the value from the unit above, and passes the result to the unit below. Input vectors are fed diagonally so that values align correctly as partial sums flow down the array.

*The multipliers produce 16-bit outputs; as values move down the columns of the array, each add produces 1 extra bit. Width is capped at 32, creating the potential for uncaught overflow.


### Accumulator Buffers
Result vectors from the MM Array are written to a software-specified address in a set of accumulator buffers. Instructions indicate whether values should be added into the value already at the address or overwrite it. MM instructions read from the Unified Buffer (UB) and write to the accumulator buffers; activate instructions read from the accumulator buffers and write to the UB.


### Weight FIFO
At scale (256x256 MACs), a full matrix of weights (a "tile") is 64KB; to avoid stalls while weights are moved from off-chip weight DRAM, a 4-entry FIFO is used to buffer tiles. It is assumed the connection to the weight DRAM is a standard DDR interface moving data in 64-byte chunks (memory controllers are currently emulated with no simulated delay, so one chunk arrives each cycle). When an MM instruction carries the "switch" flag, each MAC switches the active weight buffer as first vector of the instruction propagates through the array. Once it reaches the end of the first row, the FIFO begins feeding new weight values into the free buffers of the array. New weight values are passed down through the array each cycle until each row reaches its destination.


### Systolic Setup
Vectors are read all at once from the Unified Buffer, but must be fed diagonally into the MM Array. This is accomplished with a set of sequential buffers in a lower triangular configuration. The top value reaches the matrix immediately, the second after one cycle, the third after two, etc., so that each value reaches a MAC at the same time as the corresponding partial sum from the same source vector.


### Memory Controllers
Currently, memory controllers are emulated and have no delay. The connection to Host Memory is currently the size of one vector. The connection to the Weight DRAM uses a standard width of 64 bytes.

Because the emulated controllers can return a new value each cycle, the OpenTPU hardware simulation currently has no non-detministic delay. With a more accurate DRAM interface that may encounter dynamic delays, programs would need to either take care to schedule for the worst-case memory delay, or make use of another instruction to ensure memory operations complete before the values are required*.

*We note that the TPU "SYNC" instruction may fulfill this purpose, but is currently unimplemented on OpenTPU.


### Configuration
Unified Buffer size, Accumulator Buffer size, and the size of the MM Array can all be specified in config.py. However, the MM Array must always be square, and vectors/weights are always composed of 8-bit integers.


