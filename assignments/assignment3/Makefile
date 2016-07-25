all: matMul matMulD

matMul: main.cu matrix.cu
	nvcc main.cu matrix.cu -o matMul

matMulD: main.cu matrix.cu
	nvcc main.cu matrix.cu -o matMulD -DDEBUG=1

clean:
	rm matMul matMulD

report/HW3.pdf: report
	(cd report; python2 plotter.py; pdflatex HW3.tex)


