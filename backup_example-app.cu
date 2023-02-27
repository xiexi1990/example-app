#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__ void sum(float *x)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  float mul_temp[32];
  float result = 0;

  // #pragma unroll
  //   for (int j = 0; j < 32; ++j)
  //   {
  //     mul_temp[j] = x[j];
  //   }
  // #pragma unroll
  //   for (int j = 0; j < 32; ++j)
  //   {
  //     result += mul_temp[j];
  //   }
  float xi = x[tid];
  for (int offset = 1; offset < 32; offset *= 2)
  {
    result += __shfl_xor_sync(-1, xi, offset);
  }

  x[0] = result;
}
int main(void)
{
  int N = 32;
  float *x; // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N * sizeof(float));

  float s = 0;
  for (int i = 0; i < N; i++)
  {
    x[i] = i;
    s += x[i];
  }
  // Run kernel on 1M elements on the GPU
  sum<<<1, 32>>>(x);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  std::cout << "sum result: " << x[0] << " " << s << std::endl;
  // Free memory
  cudaFree(x);
  return 0;
}