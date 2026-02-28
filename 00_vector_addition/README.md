https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html
https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html


Goal: Learn basics and start with basic vector addition. Then you'll be read to learn matmul yourself.


### Introduction
As GPUs advanced and become more programmable, graphics pipelines emerged allowing custom code to run in parallel for 3D scenes and images. In 2006, nvidia introduced the Compute Unified Device Architecture with the goal of allowing any computational workload to take advantage of the throughput capability of GPUs, not just graphics APIs.


GPUs differ from CPUs in that they're designed for higher throughput by processing thousands of operations in parallel. CPUs have better performance for single-thread sequential tasks.


The CUDA GPU platform can be programmable with languages such as C++. There's also optimized, portable libraries such as cuBLAS, cuFFT, cuDNN, and CUTLASS that are likely better to use then reimplementing yourself.

Many frameworks and DSLs (NVIDIA Warp and OpenAI Triton) exist as well, so you likely won't need to write C++ to take advantage of CUDA.

### Programming Model
