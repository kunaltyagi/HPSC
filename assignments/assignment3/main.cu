#include <stdio.h>
#include <cuda_runtime.h>
#include "matrix.h"
//#define DEBUG

int main()
{
    int nArray[] = {100, 200, 500, 1000, 2000, 5000, 10000};
    int indexLength = sizeof(nArray)/sizeof(int);
    for (int i=0; i<indexLength; i++)
    {
        struct timeval start, end;
        gettimeofday(&start, NULL);

        int n = nArray[i];
        double **a =getmatA(n);
        double **b =getmatB(n);
        double **c =allocate(n);
        double *dev_a, *dev_b, *dev_c;
        int *dev_n;
        int size = sizeof(double)*n*n;
        int size2 = sizeof(double)*n;
        cudaMalloc( (void**)&dev_b, size );
        cudaMemcpy( dev_b, &b[0][0], size, cudaMemcpyHostToDevice );
        cudaMalloc( (void**)&dev_a, size2 );
        cudaMalloc( (void**)&dev_c, size2 );
        cudaMalloc( (void**)&dev_n,sizeof(int) );
        cudaMemcpy( dev_n, &n, sizeof(int), cudaMemcpyHostToDevice );

        for(int i=0;i<n;i++)
        {
#ifdef DEBUG
            if(i%100 ==0)
            {
                gettimeofday(&end, NULL);
                double delta = ((end.tv_sec  - start.tv_sec) * 1000000u +
                        end.tv_usec - start.tv_usec) / 1.e6;
                printf("%d,%f\n", i,delta);
            }
#endif
            cudaMemcpy( dev_a, &a[i][0], size2, cudaMemcpyHostToDevice );
            matrixMultiplication<<< n/100, 100 >>>( dev_a, dev_b, dev_c, dev_n );
            cudaMemcpy( &c[i][0], dev_c, size2, cudaMemcpyDeviceToHost );
        }
        cudaFree( dev_a );
        cudaFree( dev_b );
        cudaFree( dev_c );
        cudaFree( dev_n );

        gettimeofday(&end, NULL);
        double delta = ((end.tv_sec  - start.tv_sec) * 1000000u +
                end.tv_usec - start.tv_usec) / 1.e6;
        printf("%d,%f\n", n,delta);
#ifdef DEBUG
        printMat(a,n);
        printMat(b,n);
        printMat(c,n);
#endif
        free(&a[0][0]);
        free(&b[0][0]);
        free(&c[0][0]);

    }
return 0;
}
