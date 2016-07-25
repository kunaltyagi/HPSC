#ifndef _MATWRAPPER_H_
#define _MATWRAPPER_H_

#include "header.h"

template <template <class A, size_t B, size_t C> class MAT,
          class T,
          size_t RA,
          size_t CA,
          size_t CB>
struct MatWrapper
{
    MAT<T, RA, CA> a;
    MAT<T, CB, CA> bt;
    MAT<T, RA, CB> c;

    void init(bool allZero = false)
    {
        for (unsigned int i = 0; i < RA; ++i)
        {
            for (unsigned int j = 0; j < CA; ++j)
            {
                a[i][j] = allZero ? 0 : (i + 1) + (j + 1);
            }
        }
        for (unsigned int i = 0; i < CB; ++i)
        {
            for (unsigned int j = 0; j < CA; ++j)
            {
                bt[i][j] = allZero ? 0 : (j + 1) * (i + 1);
            }
        }
        for (unsigned int i = 0; i < RA; ++i)
        {
            for (unsigned int j = 0; j < CB; ++j)
            {
                c[i][j] = 0;
            }
        }
    }

#ifndef N
#define MULTIPLY(a_, bt_, c_, rows) MULTIPLY_DEBUG(a_, bt_, c_, rows)
#else
#define MULTIPLY(a_, bt_, c_, rows) MULTIPLY_FAST(a_, bt_, c_, rows)
#endif

#define MULTIPLY_FAST(a_, bt_, c_, rows)\
    LOOP_i(rows)\
    INNER_LOOPS(a_, bt_, c_)
#define MULTIPLY_DEBUG(a_, bt_, c_, rows)\
    LOOP_i(rows)\
    LOOP_i_info\
    INNER_LOOPS(a_, bt_, c_)

#define LOOP_i(rows)\
    for (unsigned int i = 0; i < rows; ++i)          \
    {
#define LOOP_i_info\
        std::cout << "Thread " << tid << " did row " << i << '\n';
#define INNER_LOOPS(a_, bt_, c_)\
        for (unsigned int j = 0; j < CB; ++j)        \
        {                                            \
            for (unsigned int k = 0; k < CA; ++k)    \
            {                                        \
                c_[i][j] += a_[i][k] * bt_[j][k];    \
            }                                        \
        }                                            \
    }

    void openmpMult(int chunk = 1)
    {
#ifndef N
        int tid = 0, threadNum;
#pragma omp parallel default(shared) shared(threadNum, chunk) private(tid)
#else
#pragma omp parallel default(shared) shared(chunk)
#endif
#ifndef N
        threadNum = omp_get_num_threads();
        tid = omp_get_thread_num();
        if (tid == 0)
        {
            std::cout << "Multiplying with " << threadNum << " threads\n";
        }
        else
        {
            std::cout << "Working (" << tid << ")...\n";
        }
#endif
        std::cout << "OpenMP mult: " << RA << ' ' << CA << ' ' << CB << '\n';
#pragma omp for schedule (static, chunk)
        MULTIPLY(a, bt, c, RA);
    }
    void simpleMult()
    {
#ifndef N
        std::cout<< "Multiplying with only O2 flag, no parallelization\n";
        int tid = 0;
#endif
        std::cout << "Simple mult: " << RA << ' ' << CA << ' ' << CB << '\n';
        MULTIPLY(a, bt, c, RA);
    }
    void mpiMult()
    {
        int taskId = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &taskId);
        MPI_Comm_rank(MPI_COMM_WORLD, &_numTasks);

        if (_numTasks < 2)
        {
#ifndef N
            std::cout << "Only found (" << _numTasks << "/2) MPI tasks. Quitting...\n";
#endif
            int rc = 0;
            MPI_Abort(MPI_COMM_WORLD, rc);
            std::exit(1);
        }
        if (taskId == MASTER)
        {
            _mpiMaster();
        }
        else
        {
            _mpiWorker(taskId);
        }
    }

    void cudaMult() {}
    void openclMult() {}
    void printA_Bt()
    {
        std::cout << "Matrix A:\n";
        _print<RA, CA>(a);
        std::cout << "Matrix B transposed:\n";
        _print<CB, CA>(bt);
    }
    void printC()
    {
        std::cout << "Matrix C (Result):\n";
        _print<RA, CB>(c);
    }

    protected:
    int _numTasks;
    const int MASTER = 0, FROM_MASTER = 1, FROM_WORKER = 2;
    template <size_t Row, size_t Col>
    void _print(MAT<T, Row, Col> a)
    {
        for (unsigned int i = 0; i < Row; ++i)
        {
            for (unsigned int j = 0; j < Col; ++j)
            {
                std::cout << a[i][j] << ' ';
            }
            std::cout << '\n';
        }
    }
    void _mpiMaster()
    {
#ifndef N
        std::cout << "MPI started with " << _numTasks << " tasks.\n";
#endif
        // one is master
        int workers = _numTasks - 1;
        int avgRow = RA / workers, extra  = RA % workers, offset = 0;
        unsigned int rows;
        for (int dest = 1; dest <= workers; ++dest)
        {
            rows = dest <= extra ? avgRow + 1 : avgRow;
#ifndef N
        std::cout << "Sending " << rows << " rows to task " << dest
                  << " with offset = " << offset;
#endif
            MPI_Send(&offset, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&a[offset][0], rows*CA, MPI_FLOAT, dest,
                    FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&bt[offset][0], rows*CA, MPI_FLOAT, dest,
                    FROM_MASTER, MPI_COMM_WORLD);
        }
        for (int dest = 1; dest <= workers; ++dest)
        {
            MPI_Status status;
            MPI_Recv(&offset, 1, MPI_INT, dest,
                    FROM_WORKER, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, dest,
                    FROM_WORKER, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset][0], rows*CB, MPI_FLOAT, dest,
                    FROM_WORKER, MPI_COMM_WORLD, &status);
#ifndef N
        std::cout << "Received result from task " << dest << '\n';
#endif
        }
    }
    void _mpiWorker(int tid)
    {
        MPI_Status status;
        int offset;
        unsigned int rows;
        MPI_Recv(&offset, 1, MPI_INT,
                MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT,
                MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&a, rows*CA, MPI_FLOAT,
                MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&bt, rows*CA, MPI_FLOAT,
                MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);

        MULTIPLY(a, bt, c, rows);

        MPI_Send(&offset, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&c, rows*CB, MPI_FLOAT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
    }
};

template <size_t RA, size_t CA, size_t CB>
using WrapperMatNative = MatWrapper<MatrixNative, float, RA, CA, CB>;

#endif  // _MATWRAPPER_H_
