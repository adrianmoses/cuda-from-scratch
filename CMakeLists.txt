cmake_minimum_required(VERSION 3.18)
project(cuda-from-scratch LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Default to Release for benchmarking; use -DCMAKE_BUILD_TYPE=Debug for debugging
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(01_matmul)
