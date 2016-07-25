#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdlib.h>

using namespace std;


double ** allocate(int n);
__global__ void  matrixMultiplication(double *a, double *b, double *c, int *n);
void printMat(double **a, double n);
double **getmatA(int n);
double **getmatB(int n);

