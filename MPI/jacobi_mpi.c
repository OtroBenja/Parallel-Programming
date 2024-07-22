#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void printOutput(int rows, int cols, float data[]){
    
    for (int i = 0; i < rows*cols; i++){
        printf("%f ",data[i]);
        if ((i+1)%cols == 0) printf("\n");
    }
}

void readInput(int rows, int cols, float data[]){

    int partition = rows/4;

    for (int i = 0; i < rows*cols; i++){
        if (i < cols){
            data[i] = -1.0;
        } else if (i < partition*cols){
            data[i] = 3.0;
        } else if (i < 2*partition*cols){
            data[i] = 2.0;
        } else if (i < 3*partition*cols){
            data[i] = 1.0;
        } else if (i < 4*partition*cols){
            data[i] = 0.0;
        }

        if (i >= (rows-1)*cols){
            data[i] = -1.0;
        }
    }
}

int main (int argc, char *argv[]){
    // Initialise MPI
    MPI_Init(&argc, &argv);

    // Get number of processes
    int numP;
    MPI_Comm_size(MPI_COMM_WORLD, &numP);

    // Get ID of process
    int myID;
    MPI_Comm_rank(MPI_COMM_WORLD, &myID);

    if(argc < 4){
        // Only first process prints message
        if(myID == 0){
            printf("Program should be called as ./jacobi rows cols errThreshold\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);
    float errThres = atof(argv[3]);

    if ((rows < 1) || (cols < 1)){
        // First process prints message
        if(myID == 0){
            printf("Number of rows and columns must be greater than 0.\n");
            MPI_Abort(MPI_COMM_WORLD,1);
        }
    }

    if(rows%numP){
        // First process prints message
        if(myID == 0){
            printf("Number of rows must be a multiple of number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD,1);
        }
    }

    float *data;

    if(myID == 0){
        data = (float*) malloc( rows*cols*sizeof(float));
        readInput(rows, cols, data);
    }

    // The computation is divided by rows
    int myRows = rows/numP;

    MPI_Barrier(MPI_COMM_WORLD);

    // Measure current time
    double start = MPI_Wtime();

    // Arrays for the chunk of data
    float *myData = (float*) malloc( myRows*cols*sizeof(float));
    float *buff = (float*) malloc( myRows*cols*sizeof(float)); // Auxiliary array

    // Scatter the input matrix
    MPI_Scatter(data, myRows*cols, MPI_FLOAT, myData, myRows*cols, MPI_FLOAT, 0, MPI_COMM_WORLD);
    memcpy(buff, myData, myRows*cols*sizeof(float));

    float error = errThres + 1.0;
    float myError;

    // buffers to receive boundary rows
    float *prevRow = (float*) malloc( cols*sizeof(float));
    float *nextRow = (float*) malloc( cols*sizeof(float));

    while (error > errThres){
	    if (myID > 0){
	        // Send first row to previous process
	        MPI_Send(myData, cols, MPI_FLOAT, myID-1, 0, MPI_COMM_WORLD);
	        // Receive previous row from previous process
	        MPI_Recv(prevRow, cols, MPI_FLOAT, myID-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }

	    if (myID < numP-1){
	        // Send last row to next process
	        MPI_Send(&myData[(myRows-1)*cols], cols, MPI_FLOAT, myID+1, 0, MPI_COMM_WORLD);
	        // Receive next row from next process
	        MPI_Recv(nextRow, cols, MPI_FLOAT, myID+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }

	    // Update the first row
	    if ((myID > 0) && (myRows > 1)){
	        for (int j=1; j < cols-1; j++){
	            buff[j] = 0.25f*(myData[cols+j]+myData[j-1]+myData[j+1]+prevRow[j]);
	        }
	    }

	    // Update the main block
	    for (int i=1; i < myRows-1; i++){
	        for (int j=1; j < cols-1; j++){
	            // calculate discrete Laplacian with 4-point stencil
	            buff[i*cols+j] = 0.25f*(myData[(i+1)*cols+j]+myData[i*cols+j-1]+myData[i*cols+j+1]+myData[(i-1)*cols+j]);
	        }
	    }

	    // Update the last row
	    if ((myID < numP-1) && (myRows > 1)){
	        for (int j=1; j < cols-1; j++){
	            buff[(myRows-1)*cols+j] = 0.25f*(nextRow[j]+myData[(myRows-1)*cols+j-1]+myData[(myRows-1)*cols+j+1]+myData[(myRows-2)*cols+j]);
	        }
	    }

	    // Calculate the local error
	    myError = 0.0;
	    for (int i=1; i < myRows; i++){
	        for (int j=1; j < cols-1; j++){
	            // Determine difference between data and buff
	            myError += (myData[i*cols+j]-buff[i*cols+j])*(myData[i*cols+j]-buff[i*cols+j]);
	        }
	    }

    memcpy(myData, buff, myRows*cols*sizeof(float));

    // Sum error of all processes and store in 'error' on all processes
    MPI_Allreduce(&myError, &error, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }

    // Gather final matrix on process 0 for output
    MPI_Gather(myData, myRows*cols, MPI_FLOAT, data, myRows*cols, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    // Measure current time
    double end = MPI_Wtime();

    if (myID == 0){
        printf("Time with %d processes: %f seconds.\n",numP,end-start);
        printf("Final error: %f\n",error);
        //printOutput(rows, cols, data);
        free(data);
    }

    free(myData);
    free(buff);
    free(prevRow);
    free(nextRow);

    // Terminate MPI
    MPI_Finalize();

}