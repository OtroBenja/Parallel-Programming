#include <mpi.h>
#include <stdio.h>

void main(){
    //initialize MPI
    MPI_Init(NULL,NULL);

    //get number of processes
    int numP;
    MPI_Comm_size(MPI_COMM_WORLD, &numP);

    //get rank of process
    int myId;
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);

    //every process prints Hello World
    printf("Process %d of %d: Hello World!\n",myId,numP);

    MPI_Finalize();
}