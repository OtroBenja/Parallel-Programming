#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define SAVE_RES 50
#define SAVE_ITERATION 100
#define ITERATIONS 10000
#define E 2.718281828
#define C 1.0


float* initialize_field(float p0,float r0,float d,float q,float deltaR,int maxR){
    int nR = maxR/deltaR;
    float* u = (float *)malloc(sizeof(float)*nR);

    //Calculate initial u
    for(int i=0;i<nR;i++){
        u[i] = p0*pow(i*deltaR,3)*pow(E,-pow((i*deltaR-r0)/d,q));
    }
    return u;
}

__global__ void step_kernel(float* u,float* ut,float* new_u,float* dR,float* dT){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    ut[idx] += (*dT)*C*C*(u[idx+1]-2*u[idx]+u[idx-1])/((*dR)*(*dR));
    new_u[idx] = u[idx] + ut[idx]*(*dT);
}
__global__ void swap_float_device(float* &a, float* &b){
    float* &temp = a;
    a = b;
    b = temp;
}

void swap_float_host(float* &a, float* &b){
    float* &temp = a;
    a = b;
    b = temp;
}

__constant__ float dR;
__constant__ float dT;
float** iteration(float* u,float deltaR,int maxR,int iterations,int save_iteration){
    int nR = maxR/deltaR;
    int size = sizeof(float)*nR;
    float deltaT=deltaR/5.;
    float* temp;
    float* ut    = (float*)malloc(size);
    float* new_u = (float*)malloc(size);
    float* d_u;    cudaMalloc((void **)&d_u,   size);
    float* d_ut;   cudaMalloc((void **)&d_ut,  size);
    float* dnew_u; cudaMalloc((void **)&dnew_u,size);
    float*  UHistory = (float*)malloc(sizeof(float)*(nR/SAVE_RES)*(iterations/save_iteration));
    float* UtHistory = (float*)malloc(sizeof(float)*(nR/SAVE_RES)*(iterations/save_iteration));
    float*  RHistory = (float*)malloc(sizeof(float)*(nR/SAVE_RES));
    float** hist     = (float**)malloc(sizeof(float*)*3);
    int save_count = save_iteration;
    //Calculate number of blocks nedeed
    int Nthread = 1024;
    //int Nblock = 1 + (nR-1)/Nthread;
    int Nblock = 1 + nR/Nthread;

    //Set constants for use in device
    cudaMemcpyToSymbol(dR,&deltaR,sizeof(float));
    cudaMemcpyToSymbol(dT,&deltaT,sizeof(float));

    //Set initial ut and u_tt
    for(int i=0;i<nR;i++)
         ut[i] = 0;
    
    //Copy data to device
    cudaMemcpy(d_u,u,size,cudaMemcpyHostToDevice);

    for(int i=0;i<iterations;i++){
        //Save values of u and ut
        if(save_count == save_iteration){
            printf("iteration %d\n",i);
            for(int ir=0;ir<(nR/SAVE_RES);ir++){
                 UHistory[(i/save_iteration)*(nR/SAVE_RES)+(ir)] = u[ir*SAVE_RES];
                UtHistory[(i/save_iteration)*(nR/SAVE_RES)+(ir)] = ut[ir*SAVE_RES];
            }
            save_count=0;
        }
        save_count+=1;
        
        //calculate ut = (c^2 *u_xx)*dt at boundaries
        ut[0]    += deltaT*C*C*(2*u[0]    -5*u[1]    +4*u[2]    -1*u[3]   )/(deltaR*deltaR);
        ut[nR-1] += deltaT*C*C*(2*u[nR-1] -5*u[nR-2] +4*u[nR-3] -1*u[nR-4])/(deltaR*deltaR);
        
        //calculate ut = (c^2 *u_xx)*dt and advance u with kernels
        step_kernel<<<Nblock,Nthread>>>(d_u,d_ut,dnew_u,&dR,&dT);
        //Copy data to host
        cudaMemcpy(new_u,dnew_u,size,cudaMemcpyDeviceToHost);
        //Swap data in device
        swap_float_device<<<1,1>>>(d_u,dnew_u);
        //swap_float_host(d_u,dnew_u);
        //Advance u at boundaries
        new_u[0]    = u[0] + ut[0]*deltaT;
        new_u[nR-1] = u[nR-1] + ut[nR-1]*deltaT;
        //Swap data in host
        temp = u;
        u = new_u;
        new_u = temp;
        //Copy boundary values of u to device
        cudaMemcpy(d_u,u,sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(&d_u[nR-1],&u[nR-1],sizeof(float),cudaMemcpyHostToDevice);
    }
        
    for(int ir=0;ir<(nR/SAVE_RES);ir++){
        RHistory[ir] = ir*SAVE_RES*deltaR;
    }
    hist[0] =  RHistory;
    hist[1] =  UHistory;
    hist[2] = UtHistory;
    return hist;
}

void print_data(float** hist,int iterations,int maxR,float deltaR,int nB,int nT,float totalTime){
    int print_iterations = iterations/SAVE_ITERATION;
    int printR = (maxR/deltaR)/SAVE_RES;
    //Add time to filename
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char fileName[50];
    snprintf(fileName, sizeof(fileName), "Output_%02d%02d%02d.dat", tm.tm_hour, tm.tm_min, tm.tm_sec);
    FILE* data = fopen(fileName,"w");

    //Print all parameters
    fprintf(data,"Execution type: Global GPU\n");
    fprintf(data,"Total simulation time: %lf\n",totalTime);
    fprintf(data,"R step size: %lf\n",deltaR);
    fprintf(data,"Maximum R: %d\n",maxR);
    fprintf(data,"Iterations: %d\n",iterations);
    fprintf(data,"Number of blocks: %d\n",nB);
    fprintf(data,"Number of threads: %d\n",nT);

    //Print R
    for(int ir=0;ir<(printR-1);ir++){
        fprintf(data,"%lf,",hist[0][ir]);
    }
    fprintf(data,"%lf\n",hist[0][printR-1]);
    //Print u
    for(int i=0;i<print_iterations;i++){
        for(int ir=0;ir<(printR-1);ir++){
            fprintf(data,"%lf,",hist[1][i*printR+ir]);
        }
        fprintf(data,"%lf\n",hist[1][i*printR+printR-1]);
    }
    //Print ut
    for(int i=0;i<print_iterations-1;i++){
        for(int ir=0;ir<(printR-1);ir++){
            fprintf(data,"%lf,",hist[2][i*printR+ir]);
        }
        fprintf(data,"%lf\n",hist[2][i*printR+printR-1]);
    }
    for(int ir=0;ir<(printR-1);ir++){
        fprintf(data,"%lf,",hist[2][(print_iterations-1)*printR+ir]);
    }
    fprintf(data,"%lf",hist[2][print_iterations*printR-1]);
    fclose(data);
}

int main(int argc, char* argv[]){

    //Asign host memory
    float* u;
    float** hist;
    
    //Define initial conditions
    float p0 = 0.001;
    float r0 = 20.;
    float d = 3.;
    float q = 2.;
    
    //Define simulation limits
    int iterations = ITERATIONS;
    if((argc>1) && atoi(argv[1])) iterations = atoi(argv[1]);
    int maxR = 80;
    if((argc>2) && atoi(argv[2])) maxR = atoi(argv[2]);
    float nT = 1;
    if((argc>3) && atoi(argv[3])) nT = atoi(argv[3]);
    nT = 1;
    float nB = 1;
    if((argc>4) && atoi(argv[4])) nB = atoi(argv[4]);
    nB = 1;
    float deltaR = 0.01;

    u = initialize_field(p0,r0,d,q,deltaR,maxR);

    //Pass initial conditions to iteration
    clock_t initTime = clock();
    hist = iteration(u,deltaR,maxR,iterations,SAVE_ITERATION);
    clock_t finalTime = clock();
    float  totalTime = (float)(finalTime-initTime) / CLOCKS_PER_SEC;

    //Print simulation history to a file
    int printR = (maxR/deltaR)/SAVE_RES;
    print_data(hist,iterations,maxR,deltaR,nT,nB,totalTime);

}


