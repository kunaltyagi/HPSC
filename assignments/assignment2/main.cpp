#include "matrix.h"
int main()
{
    int nArray[] = {100, 200, 500, 1000 ,2000, 10000};
    int indexLength = sizeof(nArray)/sizeof(int);
    for (int i=0; i<indexLength; i++)
    {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        int n = nArray[i];
        cout << nArray[i] << ", ";
        auto mat1 = getmatA(n);
        auto mat2 = getmatB(n);
        auto mat3 = matrixMultiplication(mat1,mat2,n,0,n);
        gettimeofday(&end, NULL);

        double delta = ((end.tv_sec  - start.tv_sec) * 1000000u +
                  end.tv_usec - start.tv_usec) / 1.e6;
        cout << delta << "\n";
#ifdef DEBUG
        printMat(mat1,n);
        printMat(mat2,n);
        printMat(mat3,n);
#endif
    }
    return 0;
}
