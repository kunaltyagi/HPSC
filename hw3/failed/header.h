#ifndef _HEADER_H_
#define _HEADER_H_

#include "mpi.h"
#include <omp.h>

#include <iostream>
#include <iterator>
#include <iomanip>
#include <array>
#include <vector>

#ifndef N
#define NRA 25  ///< rows in mat A
#define NCA 25  ///< cols in mat A
#define NCB 25  ///< cols in mat B
#endif

template <class T, size_t ROW, size_t COL>
using MatrixArray = std::array<std::array<T, COL>, ROW>;

template <class T, size_t ROW, size_t COL>
using MatrixVector = std::vector<std::vector<T>>;
// (COL, std::vector<T>(ROW))

template <class T, size_t ROW, size_t COL>
using MatrixNative = T[ROW][COL];

template <class T, std::size_t SIZE>
std::ostream& operator<<(std::ostream&o, const std::array<T, SIZE>& arr)
{
    std::copy(arr.begin(), arr.cend(), std::ostream_iterator<T>(o, " "));
    return o;
}

void init(int* argc, char*** argv)
{
#ifndef N
    std::cout << "Init MPI\n";
#endif
    MPI_Init(argc, argv);
}

void cleanup()
{
    MPI_Finalize();
#ifndef N
    std::cout << "Close MPI\n";
#endif
}

#endif  // _HEADER_H_
