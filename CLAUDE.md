# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A CUDA learning project progressing from raw kernels to production inference libraries. Follows an 8-phase roadmap (see LEARNING_ROADMAP.md) covering matmul, softmax, layernorm, attention, kernel fusion, transformer assembly, quantization, and inference engine study.

## Build System

- CMake with `enable_language(CUDA)`, one `CMakeLists.txt` per module directory
- Target CUDA 12.4
- Docker-based dev environment (see `docker/Dockerfile`)

```bash
# Build (from project root, once CMakeLists.txt exists)
mkdir -p build && cd build && cmake .. && make

# Build a single module
cd build && make <target_name>
```

## Profiling

- Kernel-level: `ncu` (Nsight Compute)
- System-level: `nsys` (Nsight Systems)

## Architecture

Each phase lives in a numbered directory (`01_matmul/`, `02_softmax/`, etc.) with progressively optimized `.cu` implementations. Shared utilities in `common/`:
- `utils.cuh` — timing macros, CUDA error checking
- `tensor.cuh` — simple tensor wrapper
- `benchmarks.cuh` — performance measurement helpers

## Correctness Approach

Always validate GPU kernels against CPU reference implementations and/or PyTorch (`torch.allclose()`).
