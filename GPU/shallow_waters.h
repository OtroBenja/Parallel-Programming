#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define A0 0.2
#define X0 1.5
#define Y0 1.0
#define SIGMA 0.1
#define DELTAR 0.01

#define ITERATIONS_DEFAULT 10000
#define NSAVE_DEFAULT 0
#define NX_DEFAULT 300
#define NY_DEFAULT 300
#define CFL_DEFAULT 50



#define E  2.718281828
#define PI 3.141592653
#define C 1.0
#define G 9.814
#define BDIM 32

float* initialize_field(float a0,float x0, float y0,float q,float deltaX, float deltaY,int Nx,int Ny){
    float* h = (float*)malloc(sizeof(float)*Nx*Ny);

    for(int y=0;y<Ny;y++){
        for(int x=0;x<Nx;x++){
            h[y*Nx+x] = 1.0+ a0*powf(E,-(powf(x*deltaX-x0,2)+powf(y*deltaY-y0,2))/q);
        }
    }
    return h;
}

#ifdef ISCUDA
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
#endif

void print_data(float** hist,char *exec_type,int iterations,int Nsave,int Nx,int Ny,float deltaX,float deltaY,float deltaT,int nB,int nT,float totalTime){
    float maxX = Nx*deltaX;
    float maxY = Ny*deltaY;
    //Add time to filename
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char metaFileName[50];
    char  binFileName[50];
    char    xFileName[50];
    char    yFileName[50];
    snprintf(metaFileName, sizeof(metaFileName), "Meta_%02d%02d%02d.dat", tm.tm_hour, tm.tm_min, tm.tm_sec);
    snprintf( binFileName, sizeof( binFileName), "Data.bin");
    snprintf(   xFileName, sizeof(   xFileName), "X.bin");
    snprintf(   yFileName, sizeof(   yFileName), "Y.bin");
    FILE* metaFile = fopen(metaFileName,"w");
    FILE*  binFile = fopen(binFileName,"wb");
    FILE*    xFile = fopen(  xFileName,"wb");
    FILE*    yFile = fopen(  yFileName,"wb");

    //Print all parameters
    fprintf(metaFile,"Execution type: %s\n", exec_type);
    fprintf(metaFile,"Total simulation time: %lf\n",totalTime);
    fprintf(metaFile,"X step size: %f\n",deltaX);
    fprintf(metaFile,"Y step size: %f\n",deltaY);
    fprintf(metaFile,"Time step size: %f\n",deltaT);
    fprintf(metaFile,"Number of X points: %d\n",Nx);
    fprintf(metaFile,"Number of Y points: %d\n",Nx);
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