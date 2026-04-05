cmake_minimum_required(VERSION 3.18)
project(cuda-from-scratch LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 120)

add_subdirectory(src/00_vector_addition)
