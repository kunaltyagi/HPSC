.phony: all
.phony: clean

all: multiply

N_ = 1000
PROC = 10
TIME = time -p

# the sed part is to make the warnings go away
MPI_COMPILE_FLAGS = $(shell mpicc --showme:compile | sed s/-I/-isystem/g)
# MPI_COMPILE_FLAGS = $(shell mpic++ --showme:compile)
MPI_LINK_FLAGS    = $(shell mpic++ --showme:link)
CC = g++

ifeq (${N},)
OPTIONS = -DN -DNRA=${N_} -DNCA=${N_} -DNCB=${N_}
endif

CXXFLAGS = -std=c++14 -Wall -Wextra -O2 ${MPI_COMPILE_FLAGS}
LIBS = -fopenmp ${MPI_LINK_FLAGS}

multiply: main.cpp header.h matwrapper.h
	$(CC) ${CXXFLAGS} -o multiply main.cpp ${OPTIONS} ${LIBS}

runCuda: multiply
	${TIME} multiply

clean:
	rm multiply
