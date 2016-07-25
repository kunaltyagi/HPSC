#!/bin/bash

echo "Single Thread"
./main > "report/singleThread.txt"
echo "OpenMp  Threads"
./oMpMain > "report/openMp.txt"
echo "MPI 2  Threads"
mpirun -np 2 ./mpi_main > "report/mpi2.txt"
echo "MPI 4  Threads"
mpirun -np 4 ./mpi_main > "report/mpi4.txt"
echo "MPI 8  Threads"
mpirun -np 8 ./mpi_main > "report/mpi8.txt"
