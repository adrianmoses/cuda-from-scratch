#include <cuda_runtime.h>
#include <stdlib.h>

// specific function will run on global device memory (GPU)
__global__
void vecAddKernel(float *A, float *B, float *C, int n) {
    // loop parallelism instead of a for-loop, each call is an "iteration"
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // i < n allows the function to support vectors of variable length.
    // where more threads are allocated then needed
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float *A_h, float *B_h, float *C_h, int n) {

    // CPU only
    // for (int i = 0; i < n; i++) {
    //     C_h[i] = A_h[i] + B_h[i];
    // }

    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    // Part 1 alloc device global (GPU) memory
    // cudaMalloc takes an address to a generic pointer and the data size
    cudaMalloc((void**) &A_d, size);
    cudaMalloc((void**) &B_d, size);
    cudaMalloc((void**) &C_d, size);

    cudaMemcpy(A_h, A_d, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_h, B_d, size, cudaMemcpyHostToDevice);

    // Part 2: call kernel function
    // The number of blocks and threads per block are defined at execution time
    // this is called execution configuation parameters
    // we want to allocate more than needed when the vector size (n) is > threads in block
    // Launch ceil(n/256.0) blocks with 256 threads each
    vecAddKernel<<<ceil(n/256.0),256>>>(A_d, B_d, C_d, n);

    // Part 3: copy d_C from GPU to CPU
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

}

int main() {
    // TODO: Memory allocation for A, B, C
    // I/O to read A, B, C, and N elements
    int N = 1024;
    int size = N * sizeof(float);
    float *A = (float *) malloc(size);
    float *B = (float *) malloc(size);
    float *C = (float *) malloc(size);

    vecAdd(A, B, C, N);

    free(A);
    free(B);
    free(C);
}
