#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ISCUDA
#include "shallow_waters.h"

__global__ void iter_shared(float *h_global, float *hu_global, float *hv_global,
                            float *h_new, float *hu_new, float *hv_new,
                            int Nx, int Ny,
                            float deltaX, float deltaY,  float deltaT){

    //Assign shared memory
    __shared__ float      h[(BDIM)*(BDIM)];
    __shared__ float     hu[(BDIM)*(BDIM)];
    __shared__ float     hv[(BDIM)*(BDIM)];
    __shared__ float  h_i05[(BDIM)*(BDIM)];
    __shared__ float hu_i05[(BDIM)*(BDIM)];
    __shared__ float hv_i05[(BDIM)*(BDIM)];
    __shared__ float  h_j05[(BDIM)*(BDIM)];
    __shared__ float hu_j05[(BDIM)*(BDIM)];
    __shared__ float hv_j05[(BDIM)*(BDIM)];

    //int x = threadIdx.x + blockIdx.x*blockDim.x;
    //int y = threadIdx.y + blockIdx.y*blockDim.y;

    int offx = threadIdx.x + blockIdx.x*blockDim.x -2*blockIdx.x; // x index offset by the overlapping boundary bewtween blocks
    int offy = threadIdx.y + blockIdx.y*blockDim.y -2*blockIdx.y; // y index offset by the overlapping boundary bewtween blocks

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //Copy global to shared memory and displace it to overlap with previous and next block
     h[(BDIM)*(ty) +tx] =  h_global[Nx*(offy) +offx];
    hu[(BDIM)*(ty) +tx] = hu_global[Nx*(offy) +offx];
    hv[(BDIM)*(ty) +tx] = hv_global[Nx*(offy) +offx];

    //Sync the block after copying the data to shared memory
    __syncthreads();

    //Calculate half-step in i+0.5
    if(ty<BDIM){
        if(tx<BDIM-1){
         h_i05[(BDIM-1)*ty+tx] = 0.5*( h[BDIM*ty+tx+1]+ h[BDIM*ty+tx  ])+0.5*deltaT*(hu[BDIM*ty+tx+1]-hu[BDIM*ty+tx])/deltaX;
        hv_i05[(BDIM-1)*ty+tx] = 0.5*(hv[BDIM*ty+tx+1]+hv[BDIM*ty+tx  ])
                         +0.5*deltaT*(hu[BDIM*ty+tx+1]*hv[BDIM*ty+tx+1]/h[BDIM*ty+tx+1]
                                     -hu[BDIM*ty+tx  ]*hv[BDIM*ty+tx  ]/h[BDIM*ty+tx  ])/deltaX;
        hu_i05[(BDIM-1)*ty+tx] = 0.5*(hu[BDIM*ty+tx+1]+hu[BDIM*ty+tx  ])
                         +0.5*deltaT*(hu[BDIM*ty+tx+1]*hu[BDIM*ty+tx+1]/h[BDIM*ty+tx+1] +0.5*G*h[BDIM*ty+tx+1]*h[BDIM*ty+tx+1]
                                     -hu[BDIM*ty+tx  ]*hu[BDIM*ty+tx  ]/h[BDIM*ty+tx  ] -0.5*G*h[BDIM*ty+tx  ]*h[BDIM*ty+tx  ])/deltaX;
        }
    }



    //Calculate half-step in j+0.5
    if(ty<BDIM-1){
        if(tx<BDIM){
             h_j05[BDIM*ty+tx] = 0.5*( h[BDIM*(ty+1)+tx]+ h[BDIM* ty   +tx])+0.5*deltaT*(hv[BDIM*(ty+1)+tx]-hv[BDIM*ty+tx])/deltaY;
            hu_j05[BDIM*ty+tx] = 0.5*(hu[BDIM*(ty+1)+tx]+hu[BDIM* ty   +tx])
                         +0.5*deltaT*(hu[BDIM*(ty+1)+tx]*hv[BDIM*(ty+1)+tx]/h[BDIM*(ty+1)+tx]
                                     -hu[BDIM* ty   +tx]*hv[BDIM* ty   +tx]/h[BDIM* ty   +tx])/deltaY;
            hv_j05[BDIM*ty+tx] = 0.5*(hv[BDIM*(ty+1)+tx]+hv[BDIM* ty   +tx])
                         +0.5*deltaT*(hv[BDIM*(ty+1)+tx]*hv[BDIM*(ty+1)+tx]/h[BDIM*(ty+1)+tx] +0.5*G*h[BDIM*(ty+1)+tx]*h[BDIM*(ty+1)+tx]
                                     -hv[BDIM* ty   +tx]*hv[BDIM* ty   +tx]/h[BDIM* ty   +tx] -0.5*G*h[BDIM* ty   +tx]*h[BDIM* ty   +tx])/deltaY;
        }
    }

    __syncthreads();

    //Calculate next step for h, h*u and h*v using the previous half-step
    //and save on global memory
    if(offx>0 && offx<Nx-1){
    if(offy>0 && offy<Ny-1){
    if(tx>0 && tx<BDIM-1){
        if(ty>0 && ty<BDIM-1){
             h_new[Nx*(offy)+offx] =  h[BDIM*ty+tx]
                         +deltaT*(hu_i05[(BDIM-1)*ty+tx  ]-hu_i05[(BDIM-1)*ty+tx-1])/deltaX
                         +deltaT*(hv_j05[BDIM*ty+tx]-hv_j05[BDIM*(ty-1)+tx])/deltaY;
            hu_new[Nx*(offy)+offx] = hu[BDIM*ty+tx]
                         +deltaT*(hu_j05[BDIM* ty   +tx]*hv_j05[BDIM* ty   +tx]/h_j05[BDIM* ty   +tx]
                                 -hu_j05[BDIM*(ty-1)+tx]*hv_j05[BDIM*(ty-1)+tx]/h_j05[BDIM*(ty-1)+tx])/deltaY
                         +deltaT*(hu_i05[(BDIM-1)*ty+tx  ]*hu_i05[(BDIM-1)*ty+tx  ]/h_i05[(BDIM-1)*ty+tx  ] +0.5*G*h_i05[(BDIM-1)*ty+tx  ]*h_i05[(BDIM-1)*ty+tx  ]
                                 -hu_i05[(BDIM-1)*ty+tx-1]*hu_i05[(BDIM-1)*ty+tx-1]/h_i05[(BDIM-1)*ty+tx-1] -0.5*G*h_i05[(BDIM-1)*ty+tx-1]*h_i05[(BDIM-1)*ty+tx-1])/deltaX;
            hv_new[Nx*(offy)+offx] = hv[BDIM*ty+tx]
                         +deltaT*(hu_i05[(BDIM-1)*ty+tx  ]*hv_i05[(BDIM-1)*ty+tx  ]/h_i05[(BDIM-1)*ty+tx  ]
                                 -hu_i05[(BDIM-1)*ty+tx-1]*hv_i05[(BDIM-1)*ty+tx-1]/h_i05[(BDIM-1)*ty+tx-1])/deltaX
                         +deltaT*(hu_j05[BDIM* ty   +tx]*hu_j05[BDIM* ty   +tx]/h_j05[BDIM* ty   +tx] +0.5*G*h_j05[BDIM* ty   +tx]*h_j05[BDIM* ty   +tx]
                                 -hu_j05[BDIM*(ty-1)+tx]*hu_j05[BDIM*(ty-1)+tx]/h_j05[BDIM*(ty-1)+tx] -0.5*G*h_j05[BDIM*(ty-1)+tx]*h_j05[BDIM*(ty-1)+tx])/deltaY;
        }
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
    float *h_new ;  cudaMalloc((void**)&h_new ,size);
    float *hu_new;  cudaMalloc((void**)&hu_new,size);
    float *hv_new;  cudaMalloc((void**)&hv_new,size);
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

    dim3 blockDim = dim3(BDIM,BDIM);
    //Define grid dimensions for iteration with global memory
    int xBlocks = (Nx-1)/BDIM+1;
    int yBlocks = (Ny-1)/BDIM+1;
    dim3 gridDim = dim3(xBlocks,yBlocks);
    //Define grid dimensions for iteration with shared memory
    int xBlocksShared = (Nx-1)/(BDIM-2)+1;
    int yBlocksShared = (Ny-1)/(BDIM-2)+1;
    dim3 gridDimShared = dim3(xBlocksShared,yBlocksShared);

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
 
        //Launch kernel for shared iteration
        iter_shared<<<gridDimShared,blockDim>>>(h_device,hu_device,hv_device,
                        h_new,hu_new,hv_new,
                        Nx,Ny,deltaX,deltaY,deltaT);

        //Swap pointers
        float *temp = h_device; h_device = h_new; h_new = temp;
        temp = hu_device; hu_device = hu_new; hu_new = temp;
        temp = hv_device; hv_device = hv_new; hv_new = temp;


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
    for(int i_save=0;i_save<Nsave;i_save++){
        for(int y=0;y<Ny-2;y++){
            for(int x=0;x<Nx-2;x++){
                H_hist[(Nx-2)*(Ny-2)*(i_save) + (Nx-2)*y +x] = H_hist_ghostzones[Nx*Ny*i_save + Nx*(y+1) +(x+1)];
            }
        }
    }


    for(int y=0;y<Ny-2;y++){
        for(int x=0;x<Nx-2;x++){
            H_hist[(Nx-2)*(Ny-2)*(Nsave) + (Nx-2)*y +x] = h[Nx*(y+1) +(x+1)];
        }
    }
    char exec_type[] = "Cuda shared memory";
    print_data(hist,exec_type,iterations,Nsave+1,Nx,Ny,deltaX,deltaY,deltaT,xBlocks*yBlocks,BDIM*BDIM,totalTime);
    printf("\tData saved to files\n");
    printf("All finished\n");

}