#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <iostream>
#include <vector>

#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

__global__ void cuMatMul(double *a, double *b, double *c, int *n);

double* allocate(int n)
{
    double* data = new double[n];  // malloc(sizeof(double)* n);
    return data;
}
double& access(double* array, int n, int i, int j)
{
    return array[i*n + j];
}
double** allocate(int m, int n)
{
    double** array = new double*[m];  // malloc(sizeof(double*) * n);
    double* data = new double[m * n];
    for (int i = 0; i < m; ++i)
    {
        array[i] = &access(data, n, i, 0);
    }
    return array;
}
void initA(double** a, int n, int m)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            a[i][j] = (i + 1) + (j + 1);
        }
    }
}
void initB(double** a, int n, int m)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            a[i][j] = (i + 1) * (j + 1);
        }
    }
}

#endif  // _MATRIX_H_
