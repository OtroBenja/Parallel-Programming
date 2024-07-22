#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

//#define N 64
#define MAX_ITERATIONS 10000
#define TOLERANCE 1.0E-12
#define BLOCKS 4
#define PROCESS 2

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
    //rho[cols*2 + cols/6   ] =  1;
    //rho[cols*2 + cols*7/12] =  1;
    //rho[cols*2 + cols/6   ] =  1;
    //rho[cols*2 + cols*7/12] =  1;
    //rho[cols*2 + cols*5/12] = -2;
    //rho[cols*2 + cols*5/6 ] = -2;
    //rho[cols*(rows/3  ) ] =  1;
    //rho[cols*(rows/3  ) ] =  1;
    //rho[cols*(rows*2/3) ] =  1;
    //rho[cols*(rows*2/3) ] =  1;
    //rho[cols*(rows/2  ) ] = -2;
    //rho[cols*(rows/2  ) ] = -2;
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

void main(int argc, char *argv[]){

    MPI_Init(&argc,&argv);
    int numberP;
    MPI_Comm_size(MPI_COMM_WORLD,&numberP);
    int myP;
    MPI_Comm_rank(MPI_COMM_WORLD,&myP);
    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);
    int chunkRows = atoi(argv[3]);
    int chunkCols = atoi(argv[4]);

    if(myP==0){
        //Check for proper input
        if(argc<5){
            printf("Input should be ./Filename Rows Columns yChunks xChunks\n"
                "- \'Rows\' must be a multiple of \'chunkRows\' and"
                "\'Columns\' must be a multiple of \'chunkColumns\'\n"
                "- \'chunkRows\'*\'chunkColumns\' must be equal or smaller than number of processes\n");
            printf("\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        //Check that chunkRows fit in Rows
        if(rows%chunkRows!=0){
            printf("Rows must be a multiple of chunkRows, instead got rows: %d and chunkRows: %d\n",rows,chunkRows);
            printf("\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        //Check that chunkCols fit in Cols
        if(cols%chunkCols!=0){
            printf("Cols must be a multiple of chunkCols, instead got cols: %d and chunkCols: %d\n",cols,chunkCols);
            printf("\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        //Check that chunkRows*ChunkCols match number of processes
        if(chunkCols*chunkRows != numberP){
            printf("Number of processes does not match number of chunks, got a total"
                "of %d chunks,""and %d processes\nPlease note that the number of chunks"
                "is calculated as \'chunkRows\'*\'chunkCols\'\n",chunkCols*chunkRows,numberP);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    //Sync all processes and register the time
    MPI_Barrier(MPI_COMM_WORLD);
    float initial_time = MPI_Wtime();

    //Initialize values of f and phi
    float *rho;
    float *phi;
    if (myP==0){
        rho = malloc(sizeof(float)*rows*cols);
        phi = malloc(sizeof(float)*rows*cols);
        functionRho(rows,cols,rho);
        for(int i=0;i<rows*cols;i++)
            phi[i] = rho[i];
        char *fname = "Rho_field";
        fprint_field(rows,cols,rho,fname);
    }

    //The whole space is divided into a grid of X and Y chunks
    int rowChunk = rows/chunkRows;
    int colChunk = cols/chunkCols;
    
    float *myChunk = malloc(sizeof(float)*rowChunk*colChunk);
    float *myChunkNew = malloc(sizeof(float)*rowChunk*colChunk);
    float *myRho = malloc(sizeof(float)*rowChunk*colChunk);
    float *tempPointer;

    //Buffers for boundaries with other chunks
    float *uRow = malloc(sizeof(float)*colChunk);
    float *dRow = malloc(sizeof(float)*colChunk);
    float *lCol = malloc(sizeof(float)*rowChunk);
    float *rCol = malloc(sizeof(float)*rowChunk);
    float *sendLCol = malloc(sizeof(float)*rowChunk);
    float *sendRCol = malloc(sizeof(float)*rowChunk);
    //Set boundaries buffer to zero
    for(int i=0;i<rowChunk;i++){
        lCol[i] = 0; 
        rCol[i] = 0; 
    }
    for(int j=0;j<colChunk;j++){
        uRow[j] = 0; 
        dRow[j] = 0; 
    }
    //Rearrenge the matrix to distribute it properly
    float *bufferPhi;
    float *bufferRho;
    if(myP==0){
        bufferPhi = malloc(sizeof(float)*rows*cols);
        bufferRho = malloc(sizeof(float)*rows*cols);

        int n=0;
        for(int pI=0; pI<chunkRows; pI++){
            for(int pJ=0; pJ<chunkCols; pJ++){
                for(int i=0; i<rowChunk; i++){
                    for(int j=0; j<colChunk; j++){
                        bufferPhi[n] = phi[cols*(rowChunk*pI +i) +colChunk*pJ +j];
                        bufferRho[n] = rho[cols*(rowChunk*pI +i) +colChunk*pJ +j];
                        n++;
                    }
                }
            }
        }
    }
    //Distribute the matrix over each chunk
    MPI_Scatter(bufferPhi, rowChunk*colChunk, MPI_FLOAT, myChunk, rowChunk*colChunk, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(bufferRho, rowChunk*colChunk, MPI_FLOAT, myRho,   rowChunk*colChunk, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //Iterate the algorithm for every chunk
    int t;
    float diff = 2.0 + TOLERANCE;
    float myDiff;
    for(t=0;t<MAX_ITERATIONS && diff>TOLERANCE;t++){
        //Share data from adjacent chunks
        //Share data with chunks below
        if(myP < (chunkRows-1)*chunkCols){
            MPI_Send(&myChunk[colChunk*(rowChunk-1)], colChunk, MPI_FLOAT, myP+chunkCols, 0, MPI_COMM_WORLD);
            MPI_Recv(dRow,                          colChunk, MPI_FLOAT, myP+chunkCols, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //Share data with chunks above
        if(myP > chunkCols-1){
            MPI_Send(myChunk, colChunk, MPI_FLOAT, myP-chunkCols, 0, MPI_COMM_WORLD);
            MPI_Recv(uRow,    colChunk, MPI_FLOAT, myP-chunkCols, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //Share data with chunks to the right
        if(myP%chunkCols < chunkCols-1){
            //Prepare column to send
            for(int i=0;i<chunkRows;i++)
                sendRCol[i] = myChunk[chunkCols*(i+1) -1];
            MPI_Send(sendRCol, rowChunk, MPI_FLOAT, myP+1, 0, MPI_COMM_WORLD);
            MPI_Recv(rCol,     rowChunk, MPI_FLOAT, myP+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //Share data with chunks to the left
        if(myP%chunkCols > 0){
            //Prepare column to send
            for(int i=0;i<chunkRows;i++)
                sendLCol[i] = myChunk[chunkCols*i];
            MPI_Send(sendLCol, rowChunk, MPI_FLOAT, myP-1, 0, MPI_COMM_WORLD);
            MPI_Recv(lCol,     rowChunk, MPI_FLOAT, myP-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        //Iterate on left and right boundaries of chunk
        for(int i=1;i<rowChunk-1;i++){
            myChunkNew[colChunk*i] = 0.25*(myChunk[colChunk*(i+1)] +myChunk[colChunk*(i-1)] 
                                +myChunk[colChunk*i+1] +lCol[i] -myRho[colChunk*i]);
            myChunkNew[colChunk*(i+1)-1] = 0.25*(myChunk[colChunk*(i+2)-1] +myChunk[colChunk*i-1] 
                                +rCol[i] +myChunk[colChunk*(i+1)-2] -myRho[colChunk*(i+1)-1]);
        }
        //Iterate on up and down boundaries of chunk
        for(int j=1;j<colChunk-1;j++){
            myChunkNew[j] = 0.25*(myChunk[colChunk+j] +uRow[j] 
                                +myChunk[j+1] +myChunk[j-1] -myRho[j]);
            myChunkNew[colChunk*(rowChunk-1)+j] = 0.25*(dRow[j] +myChunk[colChunk*(rowChunk-2)+j] 
                                +myChunk[colChunk*(rowChunk-1)+j+1] +myChunk[colChunk*(rowChunk-1)+j-1] 
                                -myRho[colChunk*(rowChunk-1)+j]);
        }
        //Iterate on all 4 corners
        myChunkNew[0] = 0.25*(myChunk[colChunk] +uRow[0]
                            +myChunk[1] +lCol[0] -myRho[0]);
        myChunkNew[colChunk-1] = 0.25*(myChunk[2*colChunk-1] +uRow[colChunk-1]
                            +rCol[0] +myChunk[colChunk-2] -myRho[colChunk-1]);
        myChunkNew[colChunk*(rowChunk-1)] = 0.25*(dRow[0] +myChunk[colChunk*(rowChunk-2)] 
                            +myChunk[colChunk*(rowChunk-1)+1] +lCol[rowChunk-1] -myRho[colChunk*(rowChunk-1)]);
        myChunkNew[(colChunk+1)*(rowChunk-1)] = 0.25*(dRow[colChunk-1] +myChunk[colChunk*(rowChunk-2)+rowChunk-1] 
                            +rCol[rowChunk-1] +myChunk[(colChunk+1)*(rowChunk-1)-1] -myRho[(colChunk+1)*(rowChunk-1)]);

        //Iterate on the interior of chunks
        for(int i=1;i<rowChunk-1;i++){
            for(int j=1;j<colChunk-1;j++){
                myChunkNew[colChunk*i+j] = 0.25*(myChunk[colChunk*(i+1)+j] +myChunk[colChunk*(i-1)+j] 
                                    +myChunk[colChunk*i+j+1] +myChunk[colChunk*i+j-1] -myRho[colChunk*i+j]);
            }
        }
        //Calculate squared error in each chunk
        myDiff = 0;
        for(int i=0;i<rowChunk;i++){
            for(int j=0;j<colChunk;j++){
                myDiff += (myChunkNew[colChunk*i+j]-myChunk[colChunk*i+j])*(myChunkNew[colChunk*i+j]-myChunk[colChunk*i+j]);
            }
        }
        //Calculate total error
        MPI_Allreduce(&myDiff, &diff, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        //Exchange new and previous chunk values
        tempPointer = myChunk;
        myChunk = myChunkNew;
        myChunkNew = tempPointer;
    }
    //Gather and rearrenge the whole matrix
    MPI_Gather(myChunk, rowChunk*colChunk, MPI_FLOAT, bufferPhi, rowChunk*colChunk, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if(myP==0){
        int n=0;
        for(int pI=0; pI<chunkRows; pI++){
            for(int pJ=0; pJ<chunkCols; pJ++){
                for(int i=0; i<rowChunk; i++){
                    for(int j=0; j<colChunk; j++){
                        phi[cols*(rowChunk*pI +i) +colChunk*pJ +j] = bufferPhi[n];
                        phi[cols*(rowChunk*pI +i) +colChunk*pJ +j] += 20;
                        n++;
                    }
                }
            }
        }
    }


    float final_time = MPI_Wtime();
    float delta_time = final_time-initial_time;

    //Save results to file
    printf("In process %d:\nTotal time: %f\nTotal iterations: %d\nFinal error: %f\n\n",myP,delta_time,t,diff);
    if(myP==0){
        char *fname = "Phi_solution";
        fprint_field(rows, cols, phi, fname);
    }

    //End MPI processes
    MPI_Finalize();
    
}