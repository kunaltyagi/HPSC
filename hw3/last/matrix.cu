#ifndef _MATRIX_CU_
#define _MATRIX_CU_

#include <cuda_runtime.h>

__global__ void cuMatMul(double* a, double* b, double* c, int* n)
// __global__ void cuMatMul(double* a, double* bt, double* c, int* n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    c[index] = 0;
    for (int i = 0; i < *n; ++i)
    {
        c[index] += a[i] * b[(*n)*i + index];
        // c[index] += a[i] * bt[i + (*n) * index];
    }
}

#endif  // _MATRIX_CU_
