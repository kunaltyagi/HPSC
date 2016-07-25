* Data about performace of assignments in respective folders.
* gprof (or nvprof) and valgrind used to gather data
* All runs after compilation with `-pg` flag and no optimisations
* valgrind used with option:
    * --tool=callgrind --simulate-wb=yes --cacheuse=yes
* Despite all efforts, nvprof simply returned:
    * ==<PID>== Profiling application: ./matMul
    * ==<PID>== Profiling result:
    * No kernels were profiles.
    * ==<PID>== API calls:
    * No API activities were profiled.
  A number of nvprof options along with nvcc options were tried but
  no result.
