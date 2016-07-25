# How to run
* in current directory, do ```$ make```.
* This creates required executable
* HW_CONFIG of the GPU is Nvidia GeForce 840M
* nvidia.txt: This contains run time of files

# Findings
* GPU is way faster than CPU. 3 orders of magnitude difference in N=10000
* Time increases linearly with problem size, indicating that the problem has been trivially parallelized into a vector multiplication from a matrix multiplication.
