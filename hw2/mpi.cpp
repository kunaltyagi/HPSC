#include "mpi.h"

#include "header.h"

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int main (int argc, char *argv[])
{
    int numtasks,              /* number of tasks in partition */
        taskid,                /* a task identifier */
        numworkers,            /* number of worker tasks */
        source,                /* task id of message source */
        dest,                  /* task id of message destination */
        mtype,                 /* message type */
        rows,                  /* rows of matrix A sent to each worker */
        averow, extra, offset, /* used to determine rows sent to each worker */
        i, j, k, rc;           /* misc */

    MatrixNative<double, NRA, NCA> a;
    MatrixNative<double, NCB, NCA> b;  // this is actually transpose(B)
    MatrixNative<double, NRA, NCB> c;

    MPI_Status status;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
    if (numtasks < 2 )
    {
#ifndef N
        std::cout << "Only found (" << numtasks << "/2) MPI tasks. Quitting...\n";
#endif
        MPI_Abort(MPI_COMM_WORLD, rc);
        std::exit(1);
    }
    numworkers = numtasks-1;


    /**************************** master task ************************************/
    if (taskid == MASTER)
    {
#ifndef N
        std::cout << "mpi_mm has started with " << numtasks << " tasks.\n"
                  << "Initializing arrays...\n";
#endif
        for (i = 0; i < NRA; i++)
            for (j = 0; j < NCA; j++)
                a[i][j] =  (i + 1) + (j + 1);
        for (i = 0; i < NCA; i++)
            for (j = 0; j < NCB; j++)
                b[j][i]=  (i + 1) * (j + 1);

        /* Send matrix data to the worker tasks */
        averow = NRA/numworkers;
        extra = NRA%numworkers;
        offset = 0;
        mtype = FROM_MASTER;
        for (dest=1; dest<=numworkers; dest++)
        {
            rows = (dest <= extra) ? averow+1 : averow;
#ifndef N
            std::cout << "Sending " << rows << " rows to task "
                      << dest << " offset = " << offset;
#endif
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&a[offset][0], rows*NCA, MPI_DOUBLE, dest, mtype,
                    MPI_COMM_WORLD);
            MPI_Send(&b, NCA*NCB, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            offset = offset + rows;
        }

        /* Receive results from worker tasks */
        mtype = FROM_WORKER;
        for (i=1; i<=numworkers; i++)
        {
            MPI_Recv(&offset, 1, MPI_INT, i, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, i, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset][0], rows*NCB, MPI_DOUBLE, i, mtype,
                    MPI_COMM_WORLD, &status);
#ifndef N
            std::cout << "Received results from task " << i << "\n";
#endif
        }

        /* Print results */
#ifndef N
        std::cout << "******************************************************\n";
        std::cout << "Result Matrix:\n";
        for (i = 0; i < NRA; i++)
        {
            std::cout << "\n";
            for (j = 0; j < NCB; j++)
                std::cout << c[i][j] << ' ';
        }
        std::cout << "\n******************************************************\n";
        std::cout << "Done.\n";
#endif
    }


    /**************************** worker task ************************************/
    if (taskid > MASTER)
    {
        mtype = FROM_MASTER;
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&a, rows*NCA, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&b, NCA*NCB, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

        for (k = 0; k < NCB; ++k)
            for (i = 0; i < rows; ++i)
            {
                c[i][k] = 0.0;
                for (j = 0; j < NCA; ++j)
                    c[i][k] += a[i][j] * b[k][j];
            }
        mtype = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&c, rows*NCB, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    }
    MPI_Finalize();
}
