#include <mpi.h>
#include "matrix.h"

int main(int argc, char **argv)
{
    int rank;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    int nArray[] = {100, 200, 500, 1000, 2000, 10000};
    int indexLength = sizeof(nArray)/sizeof(int);


    MPI_Status status;
    for (int i=0; i<indexLength; i++)
    {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        int n = nArray[i];
        auto mat1 = getmatA(n);
        auto mat2 = getmatB(n);
        int sIndex;
        int eIndex;
        if (rank == 0)
        {
            cout << nArray[i] << ", ";
            int noOfProcess;
            MPI_Comm_size(MPI_COMM_WORLD, &noOfProcess);
            for (int m=1; m<noOfProcess; m++)
            {
                MPI_Send(&(mat1[0][0]),n*n, MPI_DOUBLE,m, 0, MPI_COMM_WORLD);
                MPI_Send(&(mat2[0][0]),n*n, MPI_DOUBLE,m, 1, MPI_COMM_WORLD);
            }
            int k = n/noOfProcess;
            eIndex = 0;
            for (int m=0; m < noOfProcess-1; m++)
            {
                sIndex = m*k;
                eIndex = (m+1)*k;
                MPI_Send(&sIndex, 1, MPI_INT,m+1, 3, MPI_COMM_WORLD);
                MPI_Send(&eIndex, 1, MPI_INT,m+1, 4, MPI_COMM_WORLD);
            }
            sIndex = eIndex;
            eIndex = n;
            auto mat3 = matrixMultiplication(mat1,mat2,n,sIndex,eIndex);
            auto mat4 = allocate(n);
            for (int m=0; m < noOfProcess-1; m++)
            {
                sIndex = m*k;
                eIndex = (m+1)*k;
                MPI_Recv (&(mat4[0][0]), n*n, MPI_DOUBLE,m+1,5, MPI_COMM_WORLD, &status);
                for (int u =sIndex; u < eIndex; u++)
                {
                    for (int v=0; v < n; v++)
                        mat3[u][v] = mat4[u][v];

                }
            }

            gettimeofday(&end, NULL);
            double delta = ((end.tv_sec  - start.tv_sec) * 1000000u +
                    end.tv_usec - start.tv_usec) / 1.e6;
#ifdef DEBUG
            printMat(mat1,n);
            printMat(mat2,n);
            printMat(mat3,n);
#endif
            cout << delta << "\n";
        }
        else
        {
            MPI_Recv (&(mat1[0][0]), n*n, MPI_DOUBLE,0,0, MPI_COMM_WORLD, &status);
            MPI_Recv (&(mat2[0][0]), n*n, MPI_DOUBLE,0,1, MPI_COMM_WORLD, &status);
            MPI_Recv(&sIndex, 1, MPI_INT,0, 3, MPI_COMM_WORLD, &status);
            MPI_Recv(&eIndex, 1, MPI_INT,0, 4, MPI_COMM_WORLD, &status);
            auto mat3 = matrixMultiplication(mat1, mat2, n, sIndex, eIndex);
            MPI_Send(&(mat3[0][0]),n*n, MPI_DOUBLE,0, 5, MPI_COMM_WORLD);

        }
    }
    MPI_Finalize ();
    return 0;
}
