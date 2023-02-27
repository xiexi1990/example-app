#include <stdio.h>

__global__ void shufledown2(double* a, double *b,double *c, int N)
{
    double  temp = 2.0;
    __syncthreads();

   for (int offset = 32/2; offset > 0; offset /= 2){
       temp+=__shfl_down_sync(0xFFFFFFFF, temp, offset,32);
   }
    printf("%d %f %d \n",threadIdx.x ,temp,blockDim.x * gridDim.x);
}


int main(){
    double *a = NULL, *b = NULL, *c = NULL;
    shufledown2<<<1,64>>>(a, b, c, 0);
    cudaDeviceSynchronize();
}