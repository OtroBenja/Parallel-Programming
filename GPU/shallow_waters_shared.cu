#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define SAVE_RES 500
#define SAVE_ITERATION 10
#define ITERATIONS 100
#define PI 3.141592653
#define E  2.718281828
#define C 1.0
#define G 9.814
#define BDIM 32


float* initialize_field(float p0,float x0, float y0,float q,float deltaX, float deltaY,int Nx,int Ny){
    float* h = (float*)malloc(sizeof(float)*Nx*Ny);

    for(int y=0;y<Ny;y++){
        for(int x=0;x<Nx;x++){
            h[y*Nx+x] = 1.0+ p0*powf(E,-(pow(x*deltaX-x0,2)+pow(y*deltaY-y0,2))/q);
        }
    }
    return h;
}

__global__ void boundary_kernel(float *h, float *hu, float *hv, int Nx, int Ny){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    //Set boundary conditions
    //for(int x=0;x<Nx;x++){
    if(x<Nx){
         h[x] =  h[Nx*2 +x];
        hu[x] = -hu[Nx*2 +x];
        hv[x] = 0;//hv[Nx*2 +x];
         h[Nx*(Ny-1)+x] =  h[Nx*(Ny-3)+x];
        hu[Nx*(Ny-1)+x] = -hu[Nx*(Ny-3)+x];
        hv[Nx*(Ny-1)+x] = 0;//hv[Nx*(Ny-3)+x];
    }
    //for(int y=0;y<Ny;y++){
    if(y<Ny){
         h[Nx*y] =  h[Nx*y +2];
        hu[Nx*y] = 0;//hu[Nx*y +2];
        hv[Nx*y] = -hv[Nx*y +2];
         h[Nx*y+Nx-1] =  h[Nx*y+Nx-3];
        hu[Nx*y+Nx-1] = 0;//hu[Nx*y+Nx-3];
        hv[Nx*y+Nx-1] = -hv[Nx*y+Nx-3];
    }
}

__global__ void half_i_kernel(float *h, float *hu, float *hv,
                              float *h_i05, float *hu_i05, float *hv_i05,
                              int Nx, int Ny,
                              float deltaX, float deltaT){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

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

__global__ void half_j_kernel(float *h, float *hu, float *hv,
                              float *h_j05, float *hu_j05, float *hv_j05,
                              int Nx, int Ny,
                              float deltaY, float deltaT){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

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

__global__ void laststep_kernel(float *h, float *hu, float *hv,
                                float *h_i05, float *hu_i05, float *hv_i05,
                                float *h_j05, float *hu_j05, float *hv_j05,
                                int Nx, int Ny,
                                float deltaX, float deltaY,  float deltaT){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    
    //Calculate next step for h, h*u and h*v using the previous half step
    //for(int y=1;y<Ny-1;y++){
    //    for(int x=1;x<Nx-1;x++){
    if(x>0 && x<Nx-1){
        if(y>0 && y<Ny-1){
             h[Nx*y+x] += deltaT*(hu_i05[(Nx-1)*y+x  ]-hu_i05[(Nx-1)*y+x-1])/deltaX +deltaT*(hv_j05[Nx*y+x]-hv_j05[Nx*(y-1)+x])/deltaY;
            hu[Nx*y+x] += deltaT*(hu_j05[Nx* y   +x]*hv_j05[Nx* y   +x]/h_j05[Nx* y   +x]
                                 -hu_j05[Nx*(y-1)+x]*hv_j05[Nx*(y-1)+x]/h_j05[Nx*(y-1)+x])/deltaY
                         +deltaT*(hu_i05[(Nx-1)*y+x  ]*hu_i05[(Nx-1)*y+x  ]/h_i05[(Nx-1)*y+x  ] +0.5*G*h_i05[(Nx-1)*y+x  ]*h_i05[(Nx-1)*y+x  ]
                                 -hu_i05[(Nx-1)*y+x-1]*hu_i05[(Nx-1)*y+x-1]/h_i05[(Nx-1)*y+x-1] -0.5*G*h_i05[(Nx-1)*y+x-1]*h_i05[(Nx-1)*y+x-1])/deltaX;   
            hv[Nx*y+x] += deltaT*(hu_i05[(Nx-1)*y+x  ]*hv_i05[(Nx-1)*y+x  ]/h_i05[(Nx-1)*y+x  ]
                                 -hu_i05[(Nx-1)*y+x-1]*hv_i05[(Nx-1)*y+x-1]/h_i05[(Nx-1)*y+x-1])/deltaX
                         +deltaT*(hu_j05[Nx* y   +x]*hu_j05[Nx* y   +x]/h_j05[Nx* y   +x] +0.5*G*h_j05[Nx* y   +x]*h_j05[Nx* y   +x]
                                 -hu_j05[Nx*(y-1)+x]*hu_j05[Nx*(y-1)+x]/h_j05[Nx*(y-1)+x] -0.5*G*h_j05[Nx*(y-1)+x]*h_j05[Nx*(y-1)+x])/deltaY;
        }
    }
}

__global__ void iter_shared(float *h_global, float *hu_global, float *hv_global,
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

    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //Copy global to shared memory and displace it to overlap with previous and next block
     h[(BDIM)*(ty) +tx] =  h_global[Nx*(y-blockIdx.y) +x-blockIdx.x];
    hu[(BDIM)*(ty) +tx] = hu_global[Nx*(y-blockIdx.y) +x-blockIdx.x];
    hv[(BDIM)*(ty) +tx] = hv_global[Nx*(y-blockIdx.y) +x-blockIdx.x];

    /*
    //Copy global to shared memory of the boundary values
    if(ty == 0){
         h_shared[tx+1] =  h[Nx*(y) +x+1];
        hu_shared[tx+1] = hu[Nx*(y) +x+1];
        hv_shared[tx+1] = hv[Nx*(y) +x+1];
    }
    if(ty == BDIM-1){
         h_shared[(BDIM)*(ty+2) +tx+1] =  h[Nx*(y+2) +tx+1];
        hu_shared[(BDIM)*(ty+2) +tx+1] = hu[Nx*(y+2) +tx+1];
        hv_shared[(BDIM)*(ty+2) +tx+1] = hv[Nx*(y+2) +tx+1];
    }
    if(tx == 0){
         h_shared[(BDIM)*(ty+1)] =  h[Nx*(y+1) +x];
        hu_shared[(BDIM)*(ty+1)] = hu[Nx*(y+1) +x];
        hv_shared[(BDIM)*(ty+1)] = hv[Nx*(y+1) +x];
    }
    if(tx == BDIM-1){
         h_shared[(BDIM+2)*(ty+1) +tx+2] =  h[Nx*(y+1) +x+2];
        hu_shared[(BDIM+2)*(ty+1) +tx+2] = hu[Nx*(y+1) +x+2];
        hv_shared[(BDIM+2)*(ty+1) +tx+2] = hv[Nx*(y+1) +x+2];
    }
    */

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

    //The half-steps are independent of eachother, so there's no need to sync inbetween

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

    //Sync threads after both half-steps are done
    __syncthreads();

    //Calculate next step for h, h*u and h*v using the previous half-step
    //and save on global memory
    if(tx>0 && tx<BDIM-1){
        if(ty>0 && ty<BDIM-1){
             h_global[Nx*(y-blockIdx.y)+x-blockIdx.x] =  h[BDIM*ty+tx]
                         +deltaT*(hu_i05[(BDIM-1)*ty+tx  ]-hu_i05[(BDIM-1)*ty+tx-1])/deltaX
                         +deltaT*(hv_j05[BDIM*ty+tx]-hv_j05[BDIM*(ty-1)+tx])/deltaY;
            hu_global[Nx*(y-blockIdx.y)+x-blockIdx.x] = hu[BDIM*ty+tx]
                         +deltaT*(hu_j05[BDIM* ty   +tx]*hv_j05[BDIM* ty   +tx]/h_j05[BDIM* ty   +tx]
                                 -hu_j05[BDIM*(ty-1)+tx]*hv_j05[BDIM*(ty-1)+tx]/h_j05[BDIM*(ty-1)+tx])/deltaY
                         +deltaT*(hu_i05[(BDIM-1)*ty+tx  ]*hu_i05[(BDIM-1)*ty+tx  ]/h_i05[(BDIM-1)*ty+tx  ] +0.5*G*h_i05[(BDIM-1)*ty+tx  ]*h_i05[(BDIM-1)*ty+tx  ]
                                 -hu_i05[(BDIM-1)*ty+tx-1]*hu_i05[(BDIM-1)*ty+tx-1]/h_i05[(BDIM-1)*ty+tx-1] -0.5*G*h_i05[(BDIM-1)*ty+tx-1]*h_i05[(BDIM-1)*ty+tx-1])/deltaX;   
            hv_global[Nx*(y-blockIdx.y)+x-blockIdx.x] = hu[BDIM*ty+tx] 
                         +deltaT*(hu_i05[(BDIM-1)*ty+tx  ]*hv_i05[(BDIM-1)*ty+tx  ]/h_i05[(BDIM-1)*ty+tx  ]
                                 -hu_i05[(BDIM-1)*ty+tx-1]*hv_i05[(BDIM-1)*ty+tx-1]/h_i05[(BDIM-1)*ty+tx-1])/deltaX
                         +deltaT*(hu_j05[BDIM* ty   +tx]*hu_j05[BDIM* ty   +tx]/h_j05[BDIM* ty   +tx] +0.5*G*h_j05[BDIM* ty   +tx]*h_j05[BDIM* ty   +tx]
                                 -hu_j05[BDIM*(ty-1)+tx]*hu_j05[BDIM*(ty-1)+tx]/h_j05[BDIM*(ty-1)+tx] -0.5*G*h_j05[BDIM*(ty-1)+tx]*h_j05[BDIM*(ty-1)+tx])/deltaY;
        }
    }
}

void print_data(float** hist,int iterations,int Nsave,int Nx,int Ny,float deltaX,float deltaY,int nB,int nT,float totalTime){
    float maxX = Nx*deltaX;
    float maxY = Ny*deltaY;
    //Add time to filename
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char metaFileName[50];
    char  binFileName[50];
    char    xFileName[50];
    char    yFileName[50];
    snprintf(metaFileName, sizeof(metaFileName), "Meta.dat");
    snprintf( binFileName, sizeof( binFileName), "Data.bin");
    snprintf(   xFileName, sizeof(   xFileName), "X.bin");
    snprintf(   yFileName, sizeof(   yFileName), "Y.bin");
    FILE* metaFile = fopen(metaFileName,"w");
    FILE*  binFile = fopen(binFileName,"wb");
    FILE*    xFile = fopen(  xFileName,"wb");
    FILE*    yFile = fopen(  yFileName,"wb");

    //Print all parameters
    fprintf(metaFile,"Execution type: Sequential CPU\n");
    fprintf(metaFile,"Total simulation time: %lf\n",totalTime);
    fprintf(metaFile,"X step size: %f\n",deltaX);
    fprintf(metaFile,"Y step size: %f\n",deltaY);
    fprintf(metaFile,"Maximum X: %f\n",maxX);
    fprintf(metaFile,"Maximum Y: %f\n",maxY);
    fprintf(metaFile,"Iterations: %d\n",iterations);
    fprintf(metaFile,"Iterations saved: %d\n",Nsave);
    fprintf(metaFile,"Number of blocks: %d\n",nB);
    fprintf(metaFile,"Number of threads: %d\n",nT);

    //Print R to binary
    fwrite(hist[0],sizeof(float)*(Nx-2),1,xFile);
    fwrite(hist[1],sizeof(float)*(Ny-2),1,yFile);
    //Print data to binary
    fwrite(hist[2],sizeof(float)*(Nx-2)*(Ny-2),Nsave,binFile);
}

int main(int argc, char* argv[]){
    //Define initial conditions
    //int fType = 0;
    float p0 = 0.4;
    float x0 = 3.0;
    float y0 = 4.0;
    float q = 0.1;
    
    //Define simulation limits
    int iterations = ITERATIONS;
    if((argc>1) && atoi(argv[1])) iterations = atoi(argv[1]);
    int Nx = 100;
    int Ny = 100;
    if((argc>2) && atoi(argv[2])) Nx = atoi(argv[2]);
    if((argc>3) && atoi(argv[3])) Ny = atoi(argv[3]);
    float deltaR = 0.01;
    float deltaX = deltaR;
    float deltaY = deltaR;
    float cfl = 10;
    if((argc>4) && atoi(argv[4])) cfl = atof(argv[4]);
    float deltaT=deltaR/cfl;
    int Nsave = 0;
    if((argc>5) && atoi(argv[5])) Nsave = atoi(argv[5]);


    //Allocate memory on host
    int size = sizeof(float)*Nx*Ny;
    float *hu  = (float*)malloc(size);
    float *hv  = (float*)malloc(size);
    float *H_hist_ghostzones = (float*)malloc(sizeof(float)*Nx*Ny*Nsave);
    float *H_hist = (float*)malloc(sizeof(float)*(Nx-2)*(Ny-2)*(Nsave+1));
    float *X_hist = (float*)malloc(sizeof(float)*(Nx-2));
    float *Y_hist = (float*)malloc(sizeof(float)*(Ny-2));
    float **hist = (float**)malloc(sizeof(float*)*3);

    float* h;
    h = initialize_field(p0,x0,y0,q,deltaX,deltaY,Nx,Ny);
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

    dim3 blockDim = dim3(BDIM,BDIM);
    //Define grid dimensions for iteration with global memory
    int xBlocks = (Nx-1)/BDIM+1;
    int yBlocks = (Ny-1)/BDIM+1;
    dim3 gridDim = dim3(xBlocks,yBlocks);
    dim3 gridDim2 = dim3(2*xBlocks,yBlocks);
    //Define grid dimensions for iteration with shared memory
    int xBlocksShared = (Nx-1)/(BDIM-1)+1;
    int yBlocksShared = (Ny-1)/(BDIM-1)+1;
    dim3 gridDimShared = dim3(xBlocksShared,yBlocksShared);

    //Calculate when to save values of h
    int save_iter;
    int i_save = 0;
    if(Nsave>1) save_iter = iterations/Nsave;

    //Pass initial conditions to iteration
    clock_t initTime = clock();
    printf("iteration started\n");
    for(int i=0;i<iterations;i++){
        //Launch kernel for boundary conditions
        boundary_kernel<<<gridDim,blockDim>>>(h_device,hu_device,hv_device,Nx,Ny);
        /*
        //Launch kernels for half-step iteration
        half_i_kernel<<<gridDim,blockDim>>>(h_device,hu_device,hv_device,
                      h_i05,hu_i05,hv_i05,
                      Nx, Ny, deltaX, deltaT);
        half_j_kernel<<<gridDim,blockDim>>>(h_device,hu_device,hv_device,
                      h_j05,hu_j05,hv_j05,
                      Nx, Ny, deltaY, deltaT);

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
        */

        //Launch kernel for shared iteration
        iter_shared<<<gridDimShared,blockDim>>>(h_device,hu_device,hv_device,
                        Nx,Ny,deltaX,deltaY,deltaT);

        //Save intermediate step if necessary
        if(Nsave>0){
            if(i==save_iter*(i_save+1)+1){
                float *H_hist_idx;
                H_hist_idx = &H_hist_ghostzones[i_save*Nx*Ny];
                cudaMemcpyAsync(H_hist_idx,h_device,size,cudaMemcpyDeviceToHost);
                i_save++;
            }
        }

    }
    cudaDeviceSynchronize();
    clock_t finalTime = clock();
    float  totalTime = (float)(finalTime-initTime) / CLOCKS_PER_SEC;
    printf("iteration finished\n");

    //Copy data to host
    cudaMemcpy( h, h_device,size,cudaMemcpyDeviceToHost);
    cudaMemcpy(hu,hu_device,size,cudaMemcpyDeviceToHost);
    cudaMemcpy(hv,hv_device,size,cudaMemcpyDeviceToHost);
    
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
    print_data(hist,iterations,Nsave+1,Nx,Ny,deltaX,deltaY,xBlocks*yBlocks,BDIM*BDIM,totalTime);
    printf("\tData saved to files\n");
    printf("All finished\n");

}


