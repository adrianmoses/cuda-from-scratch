#include <stdio.h>
#define BLUR_SIZE 1

__global__
void blurKernal(unsigned char *in, unsigned char *out, int w, int h) {
   int col = blockIdx.x*blockDim.x + threadIdx.x;
   int row = blockIdx.y*blockDim.y + threadIdx.y;

   if (col < w && row < h) {
       int pixVal = 0;
       int pixels = 0;


       // get average of BLUR_SIZE x BLUR_SIZE box
       for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) {
           for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
               int curCol = col + blurCol;
               int curRow = row + blurRow;

               // Verify we have valid image pixels
               if(curRow >= 0 && curRow < h && curCol >=0 && curCol < w) {
                   pixVal += in[curRow*w + curCol];
                   ++pixels; // keep track of number of pixels in the average
               }
           }

           // write the new pixel value out
           out[row*w + col] = (pixVal/pixels);
       }
   }
}

int main() {
    // define image dimensions
    int w = 512;
    int h = 384;

    // Allocate CPU memory for the size of the image (w/ 1 byte each)
    // A 2D image is stored as a 1D array: row * width + col gives the pixel index
    int size = w * h * sizeof(unsigned char);
    unsigned char *in = (unsigned char *)malloc(size);
    unsigned char *out = (unsigned char *)malloc(size);

    // Initialize the image with some values
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            in[row * w + col] = (unsigned char)(row % 256);
        }
    }


    unsigned char *in_d, *out_d;

    //Allocate GPU memory
    cudaMalloc((void**) &in_d, size);
    cudaMalloc((void**) &out_d, size);

    cudaMemcpy(in, in_d, size, cudaMemcpyHostToDevice);

    dim3 gridSize(ceil(w/3.0), ceil(h/3.0), 1);
    dim3 blockSize(3, 3, 1);
    blurKernal<<<gridSize, blockSize>>>(in_d, out_d, w, h);

    cudaMemcpy(out, out_d, size, cudaMemcpyDeviceToHost);

    cudaFree(in_d);
    cudaFree(out_d);

    free(in);
    free(out);

}
