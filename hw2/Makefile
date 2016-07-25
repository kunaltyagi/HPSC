all: simple openMp mpi
	

# CPPFLAGS input from user
N = 1000
PROC = 10
TIME = time -p

CXXFLAGS=${CPPFLAGS} -std=c++11 -DN -DNRA=${N} -DNCA=${N} -DNCB=${N} -pg

simple: openmp.cpp header.h
	g++ -o simpleRun openmp.cpp -fopenmp -DOPENM ${CXXFLAGS}

openMp: openmp.cpp header.h
	g++ -o openMpRun openmp.cpp -fopenmp -DOPENMP ${CXXFLAGS}

mpi: mpi.cpp header.h
	mpic++ -o mpiRun mpi.cpp ${CXXFLAGS}

.phony: clean runMpi runSimple runOpenMp
clean:
	rm -f simpleRun openMpRun mpiRun
	rm -f core

runMpi: mpi
	${TIME} mpirun -np ${PROC} mpiRun

runSimple: simple
	${TIME} ./simpleRun

runOpenMp: openMpRun
	${TIME} ./openMpRun
