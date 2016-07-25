#include <omp.h>

#include "header.h"

int main (int argc, char *argv[])
{
    MatrixNative<double, NRA, NCA> a;
    MatrixNative<double, NCB, NCA> b;  // this is actually transpose(B)
    MatrixNative<double, NRA, NCB> c;

    int tid, nthreads, chunk;
    int i, j, k;    // loop iteration variables

    chunk = 80;                    /* set loop iteration chunk size */

    /*** Spawn a parallel region explicitly scoping all variables ***/
#ifdef OPENMP
#pragma omp parallel shared(a,b,c,nthreads,chunk) private(tid,i,j,k)
#endif
    {
#ifndef N
        tid = omp_get_thread_num();
        if (tid == 0)
        {
            nthreads = omp_get_num_threads();
            std::cout << "Starting matrix multiple example with " << nthreads << " threads\n"
                      << "Initializing matrices...\n";
        }
#endif
        /*** Initialize matrices ***/
#ifdef OPENMP
#pragma omp for schedule (static, chunk)
#endif
        for (i = 0; i < NRA; ++i)
            for (j = 0; j < NCA; ++j)
                a[i][j] = (i + 1) + (j + 1);
#ifdef OPENMP
#pragma omp for schedule (static, chunk)
#endif
        for (i = 0; i < NCA; ++i)
            for (j = 0; j < NCB; ++j)
                b[j][i] = (i + 1) * (j + 1);
#ifdef OPENMP
#pragma omp for schedule (static, chunk)
#endif
        for (i = 0; i < NRA; ++i)
            for (j = 0; j < NCB; ++j)
                c[i][j] = 0;

        /*** Do matrix multiply sharing iterations on outer loop ***/
        /*** Display who does which iterations for demonstration purposes ***/
#ifndef N
        std::cout << "Thread " << tid << " starting matrix multiply...\n";
#endif

#ifdef OPENMP
#pragma omp for schedule (static, chunk)
#endif
        for (i = 0; i < NRA; ++i)
        {
#ifndef N
            std::cout << "Thread " << tid << " did row " << i << '\n';
#endif
            for(j = 0; j < NCB; ++j)
            {
                // c[i][j] = 0;
                for (k = 0; k < NCA; ++k)
                {
                    // b = B'
                    c[i][j] += a[i][k] * b[j][k];
                }
            }
        }
    }   /*** End of parallel region ***/

#ifndef N
    for (i = 0; i < NRA; ++i)
    {
        for (j = 0; j < NCB; ++j)
            std::cout << c[i][j] << ' ';
        std::cout << '\n';
    }
#endif
    std::cout << c[999][999] << '\n';
    return 0;
}
