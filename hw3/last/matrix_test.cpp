#include "matrix.h"

int main()
{
    double* d = allocate(3*3);
    for (int i = 0; i < 3*3; ++i)
    {
        d[i] = i;
    }
    for (int i = 0; i < 3*3; ++i)
    {
        std::cout << d[i] << ' ';
    }
    std::cout << '\n';
    double** p = allocate(3,3);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            p[i][j] = i*3 + j;
        }
    }
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            std::cout << p[i][j] << ' ';
        }
        std::cout << '\n';
    }
}
