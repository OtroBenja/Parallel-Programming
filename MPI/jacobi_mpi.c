#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


void printOutput(int rows, int cols, float data[]){
    
    for (int i = 0; i < rows*cols; i++){
        printf("%f ",data[i]);
        if ((i+1)%cols == 0) printf("\n");
    }
}

void fprint_field(int rows, int cols, float* field, char *filename){
    char strBuff[50];
    snprintf(strBuff, sizeof(strBuff), "%s.csv", filename);
    FILE* values_file = fopen(strBuff,"w");
    for(int i=0;i<rows-1;i++){
        for(int j=0;j<cols-1;j++){
            fprintf(values_file,"%f,",field[i*cols +j]);
        }
        fprintf(values_file,"%f\n",field[i*cols +cols-1]);
    }
    for(int j=0;j<cols-1;j++)
        fprintf(values_file,"%f,",field[cols*(rows-1) +j]);
    fprintf(values_file,"%f",field[rows*cols-1]);
    fclose(values_file);
}

void fprint_log(int numP, float execTime, int numIter, float finalError, char *date){
    char strBuff[50];
    snprintf(strBuff, sizeof(strBuff), "Log_%s.txt", date);
    FILE* log_file = fopen(strBuff,"w");

    fprintf(log_file,"Iteration type: MPI 1D\nNumber of processes: %d\nTotal time: %f"
        "\nTotal iterations: %d\nFinal error: %f",numP,execTime,numIter,finalError);
}

void functionRho(int rows,int cols,float *rho){
    for(int i=0;i<rows*cols;i++){
        rho[i] = 0;
    }
    rho[cols*(rows/3  ) +cols/6   ] =  1;
    rho[cols*(rows/3  ) +cols*7/12] =  1;
    rho[cols*(rows*2/3) +cols/6   ] =  1;
    rho[cols*(rows*2/3) +cols*7/12] =  1;
    rho[cols*(rows/2  ) +cols*5/12] = -2;
    rho[cols*(rows/2  ) +cols*5/6 ] = -2;
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
    float *rho;

    if(myID == 0){
        data = (float*) malloc( rows*cols*sizeof(float));
        rho = (float*) malloc( rows*cols*sizeof(float));
        readInput(rows, cols, data);
        functionRho(rows, cols, rho);
    }

    // The computation is divided by rows
    int myRows = rows/numP;

    MPI_Barrier(MPI_COMM_WORLD);

    // Measure current time
    double start = MPI_Wtime();

    // Arrays for the chunk of data
    float *myData = (float*) malloc( myRows*cols*sizeof(float));
    float *myRho = (float*) malloc( myRows*cols*sizeof(float));
    float *buff = (float*) malloc( myRows*cols*sizeof(float)); // Auxiliary array

    // Scatter the input matrix
    MPI_Scatter(data, myRows*cols, MPI_FLOAT, myData, myRows*cols, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(rho, myRows*cols, MPI_FLOAT, myRho, myRows*cols, MPI_FLOAT, 0, MPI_COMM_WORLD);
    memcpy(buff, myData, myRows*cols*sizeof(float));

    float error = errThres + 1.0;
    float myError;

    // buffers to receive boundary rows
    float *prevRow = (float*) malloc( cols*sizeof(float));
    float *nextRow = (float*) malloc( cols*sizeof(float));

    //Register the number of iterations
    int iter = 0;
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
	            buff[j] = 0.25f*(myData[cols+j]+myData[j-1]+myData[j+1]+prevRow[j]-myRho[j]);
	        }
	    }

	    // Update the main block
	    for (int i=1; i < myRows-1; i++){
	        for (int j=1; j < cols-1; j++){
	            // calculate discrete Laplacian with 4-point stencil
	            buff[i*cols+j] = 0.25f*(myData[(i+1)*cols+j]+myData[i*cols+j-1]+myData[i*cols+j+1]+myData[(i-1)*cols+j]-myRho[i*cols+j]);
	        }
	    }

	    // Update the last row
	    if ((myID < numP-1) && (myRows > 1)){
	        for (int j=1; j < cols-1; j++){
	            buff[(myRows-1)*cols+j] = 0.25f*(nextRow[j]+myData[(myRows-1)*cols+j-1]+myData[(myRows-1)*cols+j+1]+myData[(myRows-2)*cols+j]-myRho[(myRows-1)*cols+j]);
	        }
	    }

	    // Calculate the local error
	    myError = 0.0;
	    for (int i=1; i < myRows; i++){
	        for (int j=1; j < cols-1; j++){
	            // Determine difference between data and buff
	            myError += (myData[i*cols+j]-buff[i*cols+j])*(myData[i*cols+j]-buff[i*cols+j]);
	        }
        iter++;
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
        //printf("Time with %d processes: %f seconds.\n",numP,end-start);
        //printf("Final error: %f\n",error);
        //Print final results to a file
        char *fname = "Phi_solution";
        time_t t = time(NULL);
        struct tm tm = *localtime(&t);
        char date[20];
        snprintf(date, sizeof(date), "%02d:%02d:%02d", tm.tm_hour, tm.tm_min, tm.tm_sec);
        char finalFname[40];
        snprintf(finalFname, sizeof(finalFname), "%s_%s", fname, date);
        fprint_field(rows, cols, data, finalFname);
        //Make log file with data about the execution
        fprint_log(numP,end-start,iter,error,date);
        free(data);
    }

    free(myData);
    free(buff);
    free(prevRow);
    free(nextRow);

    // Terminate MPI
    MPI_Finalize();

}
