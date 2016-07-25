# How to run
* in current directory, do ```$ make N=${matrix size}```.
* This creates required executables, run the required ones.
    * simpleRun: only serial
    * openMpRun: openMp implementation
    * mpiRun   : MPI implementation
* HW_CONFIG of the CPU is in sys_info, the data is more and more detailed as the file is scrolled downwards
* times.txt: This contains run time of files


# Findings
* OpenMP is fastest, because of smaller overhead
* Time increases as cube of problem size (apprrox, actually more than cube)
