#include <stdlib.h>
#include <stdio.h>

#define N 64
#define MAX_ITERATIONS 10000
#define TOLERANCE 1.0E-12

void function_f(float *f){
    for(int i=0;i<N*N;i++){
        f[i] = 0;
    }
    f[N*20 +10] = 1;
    f[N*20 +35] = 1;
    f[N*40 +10] = 1;
    f[N*40 +35] = 1;
    f[N*30 +25] = -2;
    f[N*30 +50] = -2;
}

void fprint_field(float* field){
    FILE* values_file;
    values_file = fopen("Phi_solution.csv","w");
    for(int i=0;i<N-1;i++){
        for(int j=0;j<N-1;j++){
            fprintf(values_file,"%f,",field[i*N +j]);
        }
        fprintf(values_file,"%f\n",field[i*N +N-1]);
    }
    for(int j=0;j<N-1;j++)
        fprintf(values_file,"%f,",field[N*(N-1) +j]);
    fprintf(values_file,"%f",field[N*N-1]);
    fclose(values_file);
}

void main(){
    float* phi = malloc(sizeof(float)*N*N);
    float* new_phi = malloc(sizeof(float)*N*N);
    float* f = malloc(sizeof(float)*N*N);
    float* temp_pointer;
    double diff = 10000 + TOLERANCE;

    //Initialize values of f
    function_f(f);

    //Set initial values of phi as f
    for(int i=0;i<N*N;i++)
        phi[i] = 0;
    
    //fix boundaries at 0
    for(int i=0;i<N;i++){
        phi[N*i] = 0; //left
        phi[N*i+N-1] = 0; //right
        phi[i] = 0; //top
        phi[N*(N-1) +i] = 0; //bottom
    }
    
    //Iterate the algorithm
    for(int t=0;t<MAX_ITERATIONS && diff>TOLERANCE;t++){
        //diff = 0;
        for(int i=1;i<N-1;i++){
            for(int j=1;j<N-1;j++){
                new_phi[N*i+j] = 0.25*(phi[N*(i+1)+j] +phi[N*(i-1)+j] +phi[N*i+j+1] +phi[N*i+j-1] -f[N*i+j]);
                diff += abs(new_phi[N*i+j]-phi[N*i+j]);
            }
        }
        //Exchange new and previous phi values
        temp_pointer = phi;
        phi = new_phi;
        new_phi = temp_pointer;

        //Calculate mean absolute difference from previous step
        //diff = diff/(N*N);
        if(t%100==0)
            printf("Iteration: %d, Difference: %f\n",t,diff);
    }

    //Save results to file
    fprint_field(phi);
    
    
}