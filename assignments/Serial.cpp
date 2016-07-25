#include <iostream>
#include <omp.h>

using namespace std;

void Print(int val, int *ptr){
	for (int i=0; i<val; i++)
		cout<<ptr[i]<<endl;
}

void Copy(int *frm, int *to, int count){
	for(int i=0; i<count; i++)
		frm[i] = to[i];
}


class vertex{
public:
	int numOfNeighbours;
	int *adjacent;
	int level;
	int idx;
	vertex(int numOfNeighbours, int *neighbours, int idx):
		numOfNeighbours(numOfNeighbours),
		adjacent(neighbours),
		idx(idx)
	{
	};
	vertex(){
	};
	int *GetAdjacent(){
		return adjacent;
	};

};



int main(){
	int xadj[12] = {0, 2,5,8,10,12,14,17,18,20,21,22};
	int adjncy[22] = {1, 2, 0, 3, 4, 0, 5, 6, 1, 7, 1, 8, 2, 8, 2, 9, 10, 3, 4, 5, 6, 6};	

	int numOfVertex = sizeof(xadj)/sizeof(*xadj) - 1 ;

	vertex *Graph;
	Graph = new vertex;
	for (int i=0; i<numOfVertex; i++){
		Graph[i] = *new vertex(xadj[i+1]-xadj[i], &adjncy[xadj[i]], i);
	}	

	int d[numOfVertex];
	for (int i=0; i<numOfVertex; i++)
		d[i] = -1;
		d[0] = 0;

	int level = 1;
	int numFS = 1;
	int numNS = xadj[1];
	int *FS = &xadj[0];
	int *NS = &adjncy[0];
	int *nNS;
	int counter = 1;
	int x;
	nNS = new int;

	while(numFS != 0){
		counter = 0;
		#pragma omp for(2)
		for (int i =0; i<numFS; i++){

			NS = Graph[FS[i]].GetAdjacent();
			numNS = Graph[FS[i]].numOfNeighbours;

			for (int j=0; j<numNS; j++){
				if (d[NS[j]] == -1){
					d[NS[j]] = level;	
					nNS[counter] = *new int (NS[j]);
					counter++;
				}
			}
		}

		numFS = counter;
		Copy(FS, nNS, numFS);
		level++;

	}
	Print(numOfVertex, d);
	return 0;
}
