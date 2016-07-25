#ifndef _HEADER_H_
#define _HEADER_H_

#include <iostream>
#include <iterator>
#include <iomanip>
#include <array>
#include <vector>

#ifndef N
#define NRA 1000                /* number of rows in matrix A */
#define NCA 1000                /* number of columns in matrix A */
#define NCB 1000                /* number of columns in matrix B */
#endif

template <class T, size_t ROW, size_t COL>
using Matrix = std::array<std::array<T, COL>, ROW>;

template <class T, size_t ROW, size_t COL>
using MatrixNative = T[ROW][COL];

template <class T, std::size_t SIZE>
std::ostream& operator<<(std::ostream& o, const std::array<T, SIZE>& arr)
{
    std::copy(arr.cbegin(), arr.cend(), std::ostream_iterator<T>(o, " "));
    return o;
}

#endif  // _HEADER_H_
