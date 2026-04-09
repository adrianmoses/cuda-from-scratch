

Goal: Learn basics and start with basic vector addition.


### Introduction
As GPUs advanced and become more programmable, graphics pipelines emerged allowing custom code to run in parallel for 3D scenes and images. In 2006, nvidia introduced the Compute Unified Device Architecture with the goal of allowing any computational workload to take advantage of the throughput capability of GPUs, not just graphics APIs.


GPUs differ from CPUs in that they're designed for higher throughput by processing thousands of operations in parallel. CPUs have better performance for single-thread sequential tasks.


The CUDA GPU platform can be programmable with languages such as C++. There's also optimized, portable libraries such as cuBLAS, cuFFT, cuDNN, and CUTLASS that are likely better to use then reimplementing yourself.

Many frameworks and DSLs (NVIDIA Warp and OpenAI Triton) exist as well, so you likely won't need to write C/C++ to take advantage of CUDA.

### Data Parallelism

Modern data applications often consist of processing data that are independent of one another. These are easy candidates for parallel processing and the core idea of data parallelism.

This is different from Task Parallelism: tasks that can be run at the same time instead of sequentially.

### Kernals and Grids

CUDA C extends C to allow code to run on a GPU (device) in addition to the CPU (host). Device code is executed via functions called kernels. Each kernel launch creates a grid of thread blocks, where each block contains many threads. Kernel launches are asynchronous — the host continues executing and can synchronize explicitly when device results are needed.


### Calling a kernel funtion

To define a kernal function use the `__global__` keyword before the function definition.

you have access to block and thread variables to provide access to each thread `threadIdx.x`, `blockIdx.x` and `blockDim.x`

you can reference one individual thread with `i = threadIdx.x + blockIdx.x * blockDim.x` This drills down to the block, row (blockDim) and "cell" (threadIdx)

to call a kernal funciton you use executive configuation parameters to define the block size (number of threads) and number of blocks.

`kernalFn<<blockNum, numThreadsPerBlockl>>()`


Sources:
- [nvidia - cuda programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html)
- [pmpp](https://amzn.eu/d/0g7AViMJ)
