#include <iostream>
#include <unistd.h>
#include <sys/time.h>
#include <stdlib.h>

using namespace std;


double ** allocate(int n);
double ** matrixMultiplication(double  **a, double **b, int n, int sIndex,
        int eIndex);
void printMat(double **a, double n);
double **getmatA(int n);
double **getmatB(int n);

