#include "matrix.h"

double ** allocate(int n)
{
    double **a;
    a = (double**)malloc(sizeof(double*)*n);
    double *data;
    data = (double*)malloc(sizeof(double)*n*n);
    for (int i=0; i<n;i++)
    {
        a[i] = &data[i*n];
    }
    return a;
}

__global__ void  matrixMultiplication(double *a, double *b, double *c, int *n)
{
    int index = blockIdx.x*blockDim.x+ threadIdx.x;
    c[index] = 0;
    for (int i=0; i<*n;i++)
    {
        c[index] += a[i]*b[(*n)*i+index ];

    }
}

void printMat(double **a, double n)
{
    for (int i=0; i<n; i++)
    {
        for (int j=0; j <n; j++)
        {
            printf("%f ",a[i][j]);
        }
        printf("\n");
    }
}

double **getmatA(int n)
{
    double ** A = allocate(n);
    for (int i=0; i<n;i++)
    {
        for (int j =0; j<n;j++)
        {
            A[i][j] =i+j+2;
        }
    }
    return A;
}

double **getmatB(int n)
{
    double** B = allocate(n);
    for (int i=0; i<n;i++)
    {
        for (int j =0; j<n;j++)
        {
            B[i][j]=(i+1)*(j+1);
        }
    }
    return B;
}
