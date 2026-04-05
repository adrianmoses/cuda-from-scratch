

Goal: Learn basics and start with basic vector addition. Then you'll be read to learn matmul yourself.


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


Sources:
- [nvidia - cuda programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html)
- [pmpp](https://amzn.eu/d/0g7AViMJ)
