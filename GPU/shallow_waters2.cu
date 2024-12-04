#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ISCUDA
#include "shallow_waters.h"

__global__ void half_ij_kernel(float *h, float *hu, float *hv,
                               float *h_i05, float *hu_i05, float *hv_i05,
                               float *h_j05, float *hu_j05, float *hv_j05,
                               int Nx, int Ny,
                               float deltaX, float deltaY, float deltaT){
    int half_id = 1;
    if(blockIdx.x < gridDim.x/2){
        half_id = 0;
    }

    int x = threadIdx.x + (blockIdx.x-half_id*gridDim.x/2)*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(half_id == 0){
        //for(int y=0;y<Ny;y++){
        //    for(int x=0;x<Nx-1;x++){
        if(y<Ny){
            if(x<Nx-1){
            h_i05[(Nx-1)*y+x] = 0.5*( h[Nx*y+x+1]+ h[Nx*y+x  ])+0.5*deltaT*(hu[Nx*y+x+1]-hu[Nx*y+x])/deltaX;
            hv_i05[(Nx-1)*y+x] = 0.5*(hv[Nx*y+x+1]+hv[Nx*y+x  ])
                        +0.5*deltaT*(hu[Nx*y+x+1]*hv[Nx*y+x+1]/h[Nx*y+x+1]
                                    -hu[Nx*y+x  ]*hv[Nx*y+x  ]/h[Nx*y+x  ])/deltaX;
            hu_i05[(Nx-1)*y+x] = 0.5*(hu[Nx*y+x+1]+hu[Nx*y+x  ])
                        +0.5*deltaT*(hu[Nx*y+x+1]*hu[Nx*y+x+1]/h[Nx*y+x+1] +0.5*G*h[Nx*y+x+1]*h[Nx*y+x+1]
                                    -hu[Nx*y+x  ]*hu[Nx*y+x  ]/h[Nx*y+x  ] -0.5*G*h[Nx*y+x  ]*h[Nx*y+x  ])/deltaX;
            }
        }
    }

    if(half_id == 1){
        //for(int y=0;y<Ny-1;y++){
        //    for(int x=0;x<Nx;x++){
        if(y<Ny-1){
            if(x<Nx){
                h_j05[Nx*y+x] = 0.5*( h[Nx*(y+1)+x]+ h[Nx* y   +x])+0.5*deltaT*(hv[Nx*(y+1)+x]-hv[Nx*y+x])/deltaY;
                hu_j05[Nx*y+x] = 0.5*(hu[Nx*(y+1)+x]+hu[Nx* y   +x])
                        +0.5*deltaT*(hu[Nx*(y+1)+x]*hv[Nx*(y+1)+x]/h[Nx*(y+1)+x]
                                    -hu[Nx* y   +x]*hv[Nx* y   +x]/h[Nx* y   +x])/deltaY;
                hv_j05[Nx*y+x] = 0.5*(hv[Nx*(y+1)+x]+hv[Nx* y   +x])
                        +0.5*deltaT*(hv[Nx*(y+1)+x]*hv[Nx*(y+1)+x]/h[Nx*(y+1)+x] +0.5*G*h[Nx*(y+1)+x]*h[Nx*(y+1)+x]
                                    -hv[Nx* y   +x]*hv[Nx* y   +x]/h[Nx* y   +x] -0.5*G*h[Nx* y   +x]*h[Nx* y   +x])/deltaY;
            }
        }
    }
}

int main(int argc, char* argv[]){
    //Define initial conditions
    float a0 = A0;
    float x0 = X0;
    float y0 = Y0;
    float q = SIGMA;
    
    //Define simulation parameters
    int iterations = ITERATIONS_DEFAULT;
    if((argc>1) && atoi(argv[1])) iterations = atoi(argv[1]);
    int Nsave = NSAVE_DEFAULT;
    if((argc>2) && atoi(argv[2])) Nsave = atoi(argv[2]);
    int Nx = NX_DEFAULT;
    int Ny = NY_DEFAULT;
    if((argc>3) && atoi(argv[3])) Nx = atoi(argv[3]);
    if((argc>4) && atoi(argv[4])) Ny = atoi(argv[4]);
    float deltaR = DELTAR;
    float deltaX = deltaR;
    float deltaY = deltaR;
    float cfl = CFL_DEFAULT;
    if((argc>5) && atoi(argv[5])) cfl = atof(argv[5]);
    float deltaT=deltaR/cfl;


    //Allocate memory on host
    int size = sizeof(float)*Nx*Ny;
    float *hu  = (float*)malloc(size);
    float *hv  = (float*)malloc(size);
    float *H_hist_ghostzones = (float*)malloc(sizeof(float)*Nx*Ny*(Nsave+1));
    float *H_hist = (float*)malloc(sizeof(float)*(Nx-2)*(Ny-2)*(Nsave+1));
    float *X_hist = (float*)malloc(sizeof(float)*(Nx-2));
    float *Y_hist = (float*)malloc(sizeof(float)*(Ny-2));
    float **hist = (float**)malloc(sizeof(float*)*3);

    float* h;
    h = initialize_field(a0,x0,y0,q,deltaX,deltaY,Nx,Ny);
    //Set initial velocities to zero
    for(int y=0;y<Ny;y++){
        for(int x=0;x<Nx;x++){
            hu[Nx*y+x] = 0;
            hv[Nx*y+x] = 0;
        }
    }
    printf("field initialized\n");

    //Allocate memory on device
    float *h_device ;  cudaMalloc((void**)&h_device ,size);
    float *hu_device;  cudaMalloc((void**)&hu_device,size);
    float *hv_device;  cudaMalloc((void**)&hv_device,size);
    cudaMemcpy( h_device, h,size,cudaMemcpyHostToDevice);
    cudaMemcpy(hu_device,hu,size,cudaMemcpyHostToDevice);
    cudaMemcpy(hv_device,hv,size,cudaMemcpyHostToDevice);

    int size_i05 = sizeof(float)*(Nx-1)*Ny;
    int size_j05 = sizeof(float)*Nx*(Ny-1);
    float *h_i05 ; cudaMalloc((void**)&h_i05 ,size_i05);
    float *hu_i05; cudaMalloc((void**)&hu_i05,size_i05);
    float *hv_i05; cudaMalloc((void**)&hv_i05,size_i05);
    float *h_j05 ; cudaMalloc((void**)&h_j05 ,size_j05);
    float *hu_j05; cudaMalloc((void**)&hu_j05,size_j05);
    float *hv_j05; cudaMalloc((void**)&hv_j05,size_j05);

    int xBlocks = (Nx-1)/BDIM+1;
    int yBlocks = (Ny-1)/BDIM+1;
    dim3 gridDim = dim3(xBlocks,yBlocks);
    dim3 gridDim2 = dim3(2*xBlocks,yBlocks);
    dim3 blockDim = dim3(BDIM,BDIM);

    //Calculate when to save values of h
    int save_iter;
    int i_save = 0;
    if(Nsave>1) save_iter = (iterations-1)/(Nsave+1) +1;;

    //Pass initial conditions to iteration
    printf("iteration started\n");
    clock_t initTime = clock();
    for(int i=0;i<iterations;i++){
        //Launch kernel for boundary conditions
        boundary_kernel<<<gridDim,blockDim>>>(h_device,hu_device,hv_device,Nx,Ny);

       //Launch kernel for half-step iteration
        half_ij_kernel<<<gridDim2,blockDim>>>(h_device,hu_device,hv_device,
                      h_i05,hu_i05,hv_i05,
                      h_j05,hu_j05,hv_j05,
                      Nx, Ny, deltaX, deltaY, deltaT);


        //Launch kernel for final step
        laststep_kernel<<<gridDim,blockDim>>>(h_device,hu_device,hv_device,
                        h_i05,hu_i05,hv_i05,
                        h_j05,hu_j05,hv_j05,
                        Nx, Ny, deltaX, deltaY, deltaT);

        //Save intermediate step if necessary
        if(Nsave>0){
            if(i==save_iter*(i_save)){
                float *H_hist_idx;
                H_hist_idx = &H_hist_ghostzones[i_save*Nx*Ny];
                cudaMemcpy(H_hist_idx,h_device,size,cudaMemcpyDeviceToHost);
                i_save++;
            }
        }

    }
    clock_t finalTime = clock();
    float  totalTime = (float)(finalTime-initTime) / CLOCKS_PER_SEC;
    printf("iteration finished\n");
    
    //Copy data to host
    float *H_hist_idx;
    H_hist_idx = &H_hist_ghostzones[Nsave*Nx*Ny];
    cudaMemcpy(H_hist_idx,h_device,size,cudaMemcpyDeviceToHost);

    //Print simulation history to a file
    printf("saving data...");
    for(int x=0;x<Nx-2;x++)
        X_hist[x] = x*deltaX;
    for(int y=0;y<Ny-2;y++)
        Y_hist[y] = y*deltaY;
    hist[0] =  X_hist;
    hist[1] =  Y_hist;
    hist[2] =  H_hist;
    
    //Reestructure data to eliminate ghostzones
    for(int i_save=0;i_save<Nsave+1;i_save++){
        for(int y=0;y<Ny-2;y++){
            for(int x=0;x<Nx-2;x++){
                H_hist[(Nx-2)*(Ny-2)*(i_save) + (Nx-2)*y +x] = H_hist_ghostzones[Nx*Ny*i_save + Nx*(y+1) +(x+1)];
            }
        }
    }

    char exec_type[] = "Cuda global memory 2";
    print_data(hist,exec_type,iterations,Nsave+1,Nx,Ny,deltaX,deltaY,deltaT,xBlocks*yBlocks,BDIM*BDIM,totalTime);
    printf("\tData saved to files\n");
    printf("All finished\n");

}