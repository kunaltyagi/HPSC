all: main oMpMain mpi_main

main: main.cpp matrix.cpp
	g++ -std=c++11 -o  main  main.cpp matrix.cpp

oMpMain: main.cpp matrix.cpp
	g++ -std=c++11  -fopenmp -o oMpMain main.cpp matrix.cpp

mpi_main: mpi_main.cpp matrix.cpp
	mpic++ -std=c++11 -o mpi_main mpi_main.cpp matrix.cpp


clean:
	rm mpi_main oMpMain main

report/HW2.pdf: report
	(cd report; python2 plotter.py; pdflatex HW2.tex)
