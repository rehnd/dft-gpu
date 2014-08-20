#include <iostream>
#include <math.h>

#include "cufft.h"

#define NX 256
#define BATCH 1

int main(int argc, char* argv[])
{
  cufftHandle plan;
  cufftComplex *data;
  cudaMalloc((void**)&data, sizeof(cufftComplex)*NX*BATCH);

  if (cudaGetLastError() != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    return;
  }
  if (cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH) != CUFFT_SUCCESS)
  {
    fprintf(stderr, "CUFFT error: Plan creation failed");
    return;
  }
  /* Note: Identical pointers to input and output arrays implies in-place transformation */

  if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS)
  {
    fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
    return;
  }
  if (cufftExecC2C(plan, data, data, CUFFT_INVERSE) != CUFFT_SUCCESS)
  {
    fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
    return;
  }
  /* Results may not be immediately available 
     so block device until all * tasks have completed  */
  if (cudaDeviceSynchronize() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to synchronize\n");
    return;
  }
  /* Divide by number of elements in data set to get back original data */

  cufftDestroy(plan);
  cudaFree(data);
  
}