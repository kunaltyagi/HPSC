#include "matrix.h"
using namespace std;

double ** allocate(int n)
{
    double **a;
    a = new double*[n];
    double *data;
    data = new double[n*n];
    for (int i=0; i<n;i++)
    {
        a[i] = &data[i*n];
    }
    return a;
}

double ** matrixMultiplication(double  **a, double **b, int n, int sIndex,
        int eIndex)
{
    double **c = allocate(n);
    #pragma omp parallel for
    for (int i=sIndex; i<eIndex; i++)
    {
        for (int j=0; j<n; j++)
        {
            double sum = 0;
            for (int k=0; k<n; k++)
            {
                sum += a[i][k]*b[k][j];
            }
            c[i][j] = sum;
        }
    }
    return c;
}

void printMat(double **a, double n)
{
    for (int i=0; i<n; i++)
    {
        for (int j=0; j <n; j++)
        {
            cout << a[i][j] <<" ";
        }
        cout << "\n";
    }
}

double **getmatA(int n)
{
    auto A = allocate(n);
    for (int i=0; i<n;i++)
    {
        for (int j =0; j<n;j++)
        {
            A[i][j] = i+j+2;
        }
    }
    return A;
}

double **getmatB(int n)
{
    auto B = allocate(n);
    for (int i=0; i<n;i++)
    {
        for (int j =0; j<n;j++)
        {
            B[i][j] = (i+1)*(j+1);
        }
    }
    return B;
}
