all: matMul

matMul: main.cu matrix.h matrix.cu
	nvcc main.cu matrix.cu -o matMul --compiler-options "-pg"

clean:
	rm matMul
