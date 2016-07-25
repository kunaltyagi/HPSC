#include "matrix.h"
#include <cuda_runtime.h>

int main()
{
    std::vector<int> sizes;
    sizes.push_back(100);
    sizes.push_back(200);
    sizes.push_back(400);
    sizes.push_back(800);
    sizes.push_back(1000);
    sizes.push_back(2000);
    sizes.push_back(4000);
    sizes.push_back(8000);
    sizes.push_back(10000);
    int threads = 100;
    int blocks = 0;
    for (int i = 0; i < sizes.size(); ++i)
    {
        int n = sizes[i];
        struct timeval start, end;
        gettimeofday(&start, NULL);

        double** a = allocate(n, n);
        double** b = allocate(n, n);
        double** c = allocate(n, n);
        initA(a,n,n);
        initB(b,n,n);
        int copySize = sizeof(double) * n;
        int allocSize = copySize * n;

        double *dev_a, *dev_b, *dev_c;
        int* dev_n;
#if A
        gettimeofday(&end, NULL);
        double del1 = (end.tv_sec - start.tv_sec) +
                      (end.tv_usec - start.tv_usec) / 1.0E6;
#endif
        cudaMalloc((void**)&dev_a, copySize);
        cudaMalloc((void**)&dev_b, allocSize);
        cudaMalloc((void**)&dev_c, copySize);
        cudaMalloc((void**)&dev_n, sizeof(int));

        cudaMemcpy(dev_b, &b[0][0], allocSize, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_n, &n, sizeof(int), cudaMemcpyHostToDevice);

#if A
        gettimeofday(&end, NULL);
        double del2 = (end.tv_sec - start.tv_sec) +
                      (end.tv_usec - start.tv_usec) / 1.0E6;
#endif

        for ( int i = 0; i < n; ++i)
        {
            blocks = n/threads;
            blocks +=  n % threads ? 1 : 0;
        cudaMemcpy(dev_a, &a[i][0], copySize, cudaMemcpyHostToDevice);
            cuMatMul<<<blocks, threads>>>(dev_a, dev_b, dev_c, dev_n);
        cudaMemcpy(&c[i][0], dev_c, copySize, cudaMemcpyDeviceToHost);
        }
#if A
        gettimeofday(&end, NULL);
        double del3 = (end.tv_sec - start.tv_sec) +
                      (end.tv_usec - start.tv_usec) / 1.0E6;
#endif


        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        cudaFree(dev_n);

        gettimeofday(&end, NULL);
#if A
        double del4 = (end.tv_sec - start.tv_sec) +
                      (end.tv_usec - start.tv_usec) / 1.0E6;
        std::cout << n << ": " << del1 << ", " << del2 << ", "
                  << del3 << ", " << del4 << ", " << del1+del2+del3+del4
#else
        double delta = (end.tv_sec - start.tv_sec) +
                       (end.tv_usec - start.tv_usec) / 1.0E6;
        std::cout << n << ", " << std::fixed << delta
#endif
                  << '\n';
    }
    return 0;
}
