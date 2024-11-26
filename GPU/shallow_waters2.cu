#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define SAVE_RES 500
#define SAVE_ITERATION 10
#define ITERATIONS 100
#define PI 3.141592653
#define E  2.718281828
#define G 9.814



float* initialize_field(float p0,float x0, float y0,float q,int deltaX,int deltaY,int maxX,int maxY){
    int Nx = maxX/deltaX;
    int Ny = maxY/deltaY;
    float* h = (float*)malloc(sizeof(float)*Nx*Ny);

    for(int y=0;y<Ny;y++){
        for(int x=0;x<Nx;x++){
            h[y*Nx+x] = 1.0+ p0*powf(E,-(pow(x*deltaX-x0,2)+pow(y*deltaY-y0,2))/q);
        }
    }
    return h;
}

__global__ void iterate_kernel(float *h, float *hu, float *hv,
                               int Nx, int Ny,
                               float dx, float dy, float dt){

    int ij,ip1,im1,jp1,jm1;
    int imm,ipp,jmm,jpp;
    float F0_i_j, F0_ip1_j, F0_im1_j;
    float F1_i_j, F1_ip1_j, F1_im1_j;
    float F2_i_j, F2_ip1_j, F2_im1_j;

    float G0_i_j, G0_i_jp1, G0_i_jm1;
    float G1_i_j, G1_i_jp1, G1_i_jm1;
    float G2_i_j, G2_i_jp1, G2_i_jm1;

    float U0_half_iphalf_j, U0_half_imhalf_j, U0_half_i_jphalf, U0_half_i_jmhalf;
    float U1_half_iphalf_j, U1_half_imhalf_j, U1_half_i_jphalf, U1_half_i_jmhalf;
    float U2_half_iphalf_j, U2_half_imhalf_j, U2_half_i_jphalf, U2_half_i_jmhalf;

    float F0_half_iphalf_j, F0_half_imhalf_j, G0_half_i_jphalf, G0_half_i_jmhalf;
    float F1_half_iphalf_j, F1_half_imhalf_j, G1_half_i_jphalf, G1_half_i_jmhalf;
    float F2_half_iphalf_j, F2_half_imhalf_j, G2_half_i_jphalf, G2_half_i_jmhalf;

    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;

    ij = j*Nx+i;
    imm = i-1;
    ipp = i+1;
    jmm = j-1;
    jpp = j+1;

    
    if (i == 0){
        imm = 1; 
        hu[ij] = -hu[ij];
        hv[ij] = 0.0;
    }
    if (i == Nx-1){
        ipp = Nx-2; 
        hu[ij] = -hu[ij];
        hv[ij] = 0.0;
    }
    if (j == 0){
        jmm = 1;
        hv[ij] = -hv[ij];
        hu[ij] = 0.0;
    }
    if (j == Ny-1){
        jpp = Ny-2;
        hv[ij] = -hv[ij];
        hu[ij] = 0.0;
    }

    ip1 = j*Nx+(ipp);
    im1 = j*Nx+(imm);
    jp1 = (jpp)*Nx+i;
    jm1 = (jmm)*Nx+i;

    F0_i_j = hu[ij];
    F1_i_j = hu[ij]*hu[ij]/ h[ij] + 0.5*G* h[ij]* h[ij];
    F2_i_j = hu[ij]*hv[ij]/ h[ij];

    G0_i_j = hv[ij];
    G1_i_j = hu[ij]*hv[ij]/ h[ij];
    G2_i_j = hv[ij]*hv[ij]/ h[ij] + 0.5*G* h[ij]* h[ij];

    F0_ip1_j = hu[ip1];
    F0_im1_j = hu[im1];

    F1_ip1_j = hu[ip1]*hu[ip1]/ h[ip1] + 0.5*G* h[ip1]* h[ip1];
    F1_im1_j = hu[im1]*hu[im1]/ h[im1] + 0.5*G* h[im1]* h[im1];

    F2_ip1_j = hu[ip1]*hv[ip1]/ h[ip1];
    F2_im1_j = hu[im1]*hv[im1]/ h[im1];

    G0_i_jp1 = hv[jp1];
    G0_i_jm1 = hv[jm1];

    G1_i_jp1 = hu[jp1]*hv[jp1]/ h[jp1];
    G1_i_jm1 = hu[jm1]*hv[jm1]/ h[jm1];

    G2_i_jp1 = hv[jp1]*hv[jp1]/ h[jp1] + 0.5*G* h[jp1]* h[jp1];
    G2_i_jm1 = hv[jm1]*hv[jm1]/ h[jm1] + 0.5*G* h[jm1]* h[jm1];

    U0_half_iphalf_j = 0.5*( h[ip1] +  h[ij]) - (dt/(2.0*dx))*(F0_ip1_j - F0_i_j);
    U0_half_imhalf_j = 0.5*( h[im1] +  h[ij]) - (dt/(2.0*dx))*(F0_i_j - F0_im1_j);

    U0_half_i_jphalf = 0.5*( h[jp1] +  h[ij]) - (dt/(2.0*dy))*(G0_i_jp1 - G0_i_j);
    U0_half_i_jmhalf = 0.5*( h[jm1] +  h[ij]) - (dt/(2.0*dy))*(G0_i_j - G0_i_jm1);

    U1_half_iphalf_j = 0.5*(hu[ip1] + hu[ij]) - (dt/(2.0*dx))*(F1_ip1_j - F1_i_j);
    U1_half_imhalf_j = 0.5*(hu[im1] + hu[ij]) - (dt/(2.0*dx))*(F1_i_j - F1_im1_j);

    U1_half_i_jphalf = 0.5*(hu[jp1] + hu[ij]) - (dt/(2.0*dy))*(G1_i_jp1 - G1_i_j);
    U1_half_i_jmhalf = 0.5*(hu[jm1] + hu[ij]) - (dt/(2.0*dy))*(G1_i_j - G1_i_jm1);

    U2_half_iphalf_j = 0.5*(hv[ip1] + hv[ij]) - (dt/(2.0*dx))*(F2_ip1_j - F2_i_j);
    U2_half_imhalf_j = 0.5*(hv[im1] + hv[ij]) - (dt/(2.0*dx))*(F2_i_j - F2_im1_j);

    U2_half_i_jphalf = 0.5*(hv[jp1] + hv[ij]) - (dt/(2.0*dy))*(G2_i_jp1 - G2_i_j);
    U2_half_i_jmhalf = 0.5*(hv[jm1] + hv[ij]) - (dt/(2.0*dy))*(G2_i_j - G2_i_jm1);

    F0_half_iphalf_j = U1_half_iphalf_j; G0_half_i_jphalf = U2_half_i_jphalf;
    F0_half_imhalf_j = U1_half_imhalf_j; G0_half_i_jmhalf = U2_half_i_jmhalf;

    F1_half_iphalf_j = U1_half_iphalf_j*U1_half_iphalf_j/U0_half_iphalf_j + 0.5*G*U0_half_iphalf_j*U0_half_iphalf_j; G1_half_i_jphalf = U1_half_i_jphalf*U2_half_i_jphalf/U0_half_i_jphalf;
    F1_half_imhalf_j = U1_half_imhalf_j*U1_half_imhalf_j/U0_half_imhalf_j + 0.5*G*U0_half_imhalf_j*U0_half_imhalf_j; G1_half_i_jmhalf = U1_half_i_jmhalf*U2_half_i_jmhalf/U0_half_i_jmhalf;

    F2_half_iphalf_j = U1_half_iphalf_j*U2_half_iphalf_j/U0_half_iphalf_j; G2_half_i_jphalf = U2_half_i_jphalf*U2_half_i_jphalf/U0_half_i_jphalf + 0.5*G*U0_half_i_jphalf*U0_half_i_jphalf;
    F2_half_imhalf_j = U1_half_imhalf_j*U2_half_imhalf_j/U0_half_imhalf_j; G2_half_i_jmhalf = U2_half_i_jmhalf*U2_half_i_jmhalf/U0_half_i_jmhalf + 0.5*G*U0_half_i_jmhalf*U0_half_i_jmhalf;

     h[ij] =  h[ij] - (dt/dx)*(F0_half_iphalf_j - F0_half_imhalf_j) - (dt/dy)*(G0_half_i_jphalf - G0_half_i_jmhalf);
    hu[ij] = hu[ij] - (dt/dx)*(F1_half_iphalf_j - F1_half_imhalf_j) - (dt/dy)*(G1_half_i_jphalf - G1_half_i_jmhalf);
    hv[ij] = hv[ij] - (dt/dx)*(F2_half_iphalf_j - F2_half_imhalf_j) - (dt/dy)*(G2_half_i_jphalf - G2_half_i_jmhalf);
}


__global__ void main_kernel(float *h, float *hu, float *hv,
                               int iterations, int Nx, int Ny,
                               float deltaX, float deltaY, float deltaT){//TO DO: Assign dX, dY, dT, Nx, and Ny as constant memory
    int xBlocks = (Nx-1)/32+1;
    int yBlocks = (Ny-1)/32+1;

    for(int i=0;i<iterations;i++){
        
        //Launch kernel for one iteration
        iterate_kernel<<<dim3(xBlocks,yBlocks),dim3(32,32)>>>(h,hu,hv,Nx,Ny,deltaX,deltaY,deltaT);

        //Wait for previous kernel here

    }

}

void print_data(float** hist,int iterations,float maxX,float maxY,float deltaR,int nB,int nT,float totalTime){
    float deltaX = deltaR;
    float deltaY = deltaR;
    int Nx = maxX/deltaX;
    int Ny = maxY/deltaY;
    int print_iterations = iterations/SAVE_ITERATION;
    //Add time to filename
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char metaFileName[50];
    char  binFileName[50];
    char    xFileName[50];
    char    yFileName[50];
    //snprintf(metaFileName, sizeof(metaFileName), "Meta_%02d%02d%02d.dat", tm.tm_hour, tm.tm_min, tm.tm_sec);
    //snprintf( binFileName, sizeof( binFileName), "Data_%02d%02d%02d.bin", tm.tm_hour, tm.tm_min, tm.tm_sec);
    //snprintf(   xFileName, sizeof(   xFileName), "X_%02d%02d%02d.bin", tm.tm_hour, tm.tm_min, tm.tm_sec);
    //snprintf(   yFileName, sizeof(   yFileName), "Y_%02d%02d%02d.bin", tm.tm_hour, tm.tm_min, tm.tm_sec);
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
    fprintf(metaFile,"R step size: %f\n",deltaR);
    fprintf(metaFile,"Maximum X: %f\n",maxX);
    fprintf(metaFile,"Maximum Y: %f\n",maxY);
    fprintf(metaFile,"Iterations: %d\n",iterations);
    fprintf(metaFile,"Number of blocks: %d\n",nB);
    fprintf(metaFile,"Number of threads: %d\n",nT);

    //Print R to binary
    fwrite(hist[0],sizeof(float)*(Nx-2),1,xFile);
    fwrite(hist[1],sizeof(float)*(Ny-2),1,yFile);
    //Print data to binary
    fwrite(hist[2],sizeof(float)*(Nx-2)*(Ny-2)*(iterations/SAVE_ITERATION),1,binFile);
}

int main(int argc, char* argv[]){
    
    //Define initial conditions
    int fType = 0;
    float p0 = 0.4;
    float x0 = 10.0;
    float y0 = 25.0;
    float q = 1.5;
    
    //Define simulation limits
    int iterations = ITERATIONS;
    if((argc>1) && atoi(argv[1])) iterations = atoi(argv[1]);
    float maxX = 20;
    float maxY = 30;
    if((argc>2) && atoi(argv[2])) maxX = atoi(argv[2]);
    if((argc>3) && atoi(argv[3])) maxY = atoi(argv[3]);
    float deltaR = 0.01;

    float deltaX = deltaR;
    float deltaY = deltaR;
    int Nx = maxX/deltaX;
    int Ny = maxY/deltaY;
    float g = 9.8;
    int size = sizeof(float)*Ny;
    float deltaT=deltaR/10.0;
    float* h;
    float *hu  = (float*)malloc(size);
    float *hv  = (float*)malloc(size);
    float *H_hist = (float*)malloc(sizeof(float)*(Nx-2)*(Ny-2)*(iterations/SAVE_ITERATION));
    float *X_hist = (float*)malloc(sizeof(float)*(Nx-2));
    float *Y_hist = (float*)malloc(sizeof(float)*(Ny-2));
    float **hist = (float**)malloc(sizeof(float*)*3);
    int save_count = SAVE_ITERATION;

    h = initialize_field(p0,x0,y0,q,deltaX,deltaY,maxX,maxY);
    //Set initial velocities to zero
    for(int y=0;y<Ny;y++){
        for(int x=0;x<Nx;x++){
            hu[Nx*y+x] = 0;
            hv[Nx*y+x] = 0;
        }
    }
    printf("field initialized\n");

    //Allocate and copy memory to kernel
    float *h_device ;  cudaMalloc((void**)&h_device ,size);
    float *u_device ;  cudaMalloc((void**)&u_device ,size);
    float *v_device ;  cudaMalloc((void**)&v_device ,size);
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

    //Pass initial conditions to iteration
    clock_t initTime = clock();
    printf("iteration started\n");
    main_kernel<<<1,1>>>(h,hu,hv,iterations,Nx,Ny,deltaX,deltaY,deltaT);
    clock_t finalTime = clock();
    float  totalTime = (float)(finalTime-initTime) / CLOCKS_PER_SEC;
    printf("iteration finished\n");

    //Print simulation history to a file
    printf("saving data...");
    //print_data(hist,iterations,maxX,maxY,deltaR,nT,nB,totalTime);
    printf("\tData saved to files\n");
    printf("All finished\n");

}


