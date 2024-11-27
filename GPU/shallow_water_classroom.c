#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define SAVE_ITERATION 10

void output_height(float *U0, int N){

  int ij;

  for (int j=0; j < N; j++){
    for (int i=0; i < N; i++){

      ij = j*N + i;

      printf("%f ",U0[ij]);
    }
    printf("\n");
  }

}

void write_data(float* U0, int N, int t){

  FILE* datafile;

  char filename[20];

  sprintf(filename, "h_%05d.dat", t);
  datafile = fopen(filename,"w");
  fwrite(&(U0[0]),sizeof(float),N*N,datafile);
  fclose(datafile);

}

void initialise_fields(float *U0, float *U1, float *U2, int N, float dx, float x0, float y0, float sigma, float A){

  int ij;

  for (int j = 0; j < N; j++){
    for (int i = 0; i < N; i++){
      
      ij = j*N + i;

      U0[ij] = 1.0+A*exp(-0.5*((x0+dx*i)*(x0+dx*i) + (y0+dx*j)*(y0+dx*j))/sigma);
      U1[ij] = 0.0;
      U2[ij] = 0.0;

    }
  }

}

void evolve_fields(float *U0, float *U1, float *U2, float *U0_np1, float *U1_np1, float *U2_np1, 
                   int N, float dx, float dt, float g){

  // Implementation of Lax-Wendroff scheme (intermediate flux calculation)
  // for the shallow water equations. NOT TESTED!!

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


  for (int j=0; j < N; j++){
    for (int i=0; i < N; i++){

      ij = j*N+i;
      imm = i-1;
      ipp = i+1;
      jmm = j-1;
      jpp = j+1;

      // boundary conditions implemented with index logic (my preferred trick)
      if (i == 0){
        imm = 1; // reflective condition in x -> use the first interior point
        U1[ij] = -U1[ij];
        U2[ij] = 0.0;
      }
      if (i == N-1){
        ipp = N-2; // reflective condition in x -> use last interior point
        U1[ij] = -U1[ij];
        U2[ij] = 0.0;
      }
      if (j == 0){
        jmm = 1; // reflection
        U2[ij] = -U2[ij];
        U1[ij] = 0.0;
      }
      if (j == N-1){
        jpp = N-2; // reflection
        U2[ij] = -U2[ij];
        U1[ij] = 0.0;
      }

      ip1 = j*N+(ipp);
      im1 = j*N+(imm);
      jp1 = (jpp)*N+i;
      jm1 = (jmm)*N+i;

      F0_i_j = U1[ij];
      F1_i_j = U1[ij]*U1[ij]/U0[ij] + 0.5*g*U0[ij]*U0[ij];
      F2_i_j = U1[ij]*U2[ij]/U0[ij];

      G0_i_j = U2[ij];
      G1_i_j = U1[ij]*U2[ij]/U0[ij];
      G2_i_j = U2[ij]*U2[ij]/U0[ij] + 0.5*g*U0[ij]*U0[ij];

      F0_ip1_j = U1[ip1];
      F0_im1_j = U1[im1];

      F1_ip1_j = U1[ip1]*U1[ip1]/U0[ip1] + 0.5*g*U0[ip1]*U0[ip1];
      F1_im1_j = U1[im1]*U1[im1]/U0[im1] + 0.5*g*U0[im1]*U0[im1];

      F2_ip1_j = U1[ip1]*U2[ip1]/U0[ip1];
      F2_im1_j = U1[im1]*U2[im1]/U0[im1];

      G0_i_jp1 = U2[jp1];
      G0_i_jm1 = U2[jm1];

      G1_i_jp1 = U1[jp1]*U2[jp1]/U0[jp1];
      G1_i_jm1 = U1[jm1]*U2[jm1]/U0[jm1];

      G2_i_jp1 = U2[jp1]*U2[jp1]/U0[jp1] + 0.5*g*U0[jp1]*U0[jp1];
      G2_i_jm1 = U2[jm1]*U2[jm1]/U0[jm1] + 0.5*g*U0[jm1]*U0[jm1];

      U0_half_iphalf_j = 0.5*(U0[ip1] + U0[ij]) - (dt/(2.0*dx))*(F0_ip1_j - F0_i_j);
      U0_half_imhalf_j = 0.5*(U0[im1] + U0[ij]) - (dt/(2.0*dx))*(F0_i_j - F0_im1_j);

      U0_half_i_jphalf = 0.5*(U0[jp1] + U0[ij]) - (dt/(2.0*dx))*(G0_i_jp1 - G0_i_j);
      U0_half_i_jmhalf = 0.5*(U0[jm1] + U0[ij]) - (dt/(2.0*dx))*(G0_i_j - G0_i_jm1);

      U1_half_iphalf_j = 0.5*(U1[ip1] + U1[ij]) - (dt/(2.0*dx))*(F1_ip1_j - F1_i_j);
      U1_half_imhalf_j = 0.5*(U1[im1] + U1[ij]) - (dt/(2.0*dx))*(F1_i_j - F1_im1_j);

      U1_half_i_jphalf = 0.5*(U1[jp1] + U1[ij]) - (dt/(2.0*dx))*(G1_i_jp1 - G1_i_j);
      U1_half_i_jmhalf = 0.5*(U1[jm1] + U1[ij]) - (dt/(2.0*dx))*(G1_i_j - G1_i_jm1);

      U2_half_iphalf_j = 0.5*(U2[ip1] + U2[ij]) - (dt/(2.0*dx))*(F2_ip1_j - F2_i_j);
      U2_half_imhalf_j = 0.5*(U2[im1] + U2[ij]) - (dt/(2.0*dx))*(F2_i_j - F2_im1_j);

      U2_half_i_jphalf = 0.5*(U2[jp1] + U2[ij]) - (dt/(2.0*dx))*(G2_i_jp1 - G2_i_j);
      U2_half_i_jmhalf = 0.5*(U2[jm1] + U2[ij]) - (dt/(2.0*dx))*(G2_i_j - G2_i_jm1);

      F0_half_iphalf_j = U1_half_iphalf_j; G0_half_i_jphalf = U2_half_i_jphalf;
      F0_half_imhalf_j = U1_half_imhalf_j; G0_half_i_jmhalf = U2_half_i_jmhalf;

      F1_half_iphalf_j = U1_half_iphalf_j*U1_half_iphalf_j/U0_half_iphalf_j + 0.5*g*U0_half_iphalf_j*U0_half_iphalf_j; G1_half_i_jphalf = U1_half_i_jphalf*U2_half_i_jphalf/U0_half_i_jphalf;
      F1_half_imhalf_j = U1_half_imhalf_j*U1_half_imhalf_j/U0_half_imhalf_j + 0.5*g*U0_half_imhalf_j*U0_half_imhalf_j; G1_half_i_jmhalf = U1_half_i_jmhalf*U2_half_i_jmhalf/U0_half_i_jmhalf;

      F2_half_iphalf_j = U1_half_iphalf_j*U2_half_iphalf_j/U0_half_iphalf_j; G2_half_i_jphalf = U2_half_i_jphalf*U2_half_i_jphalf/U0_half_i_jphalf + 0.5*g*U0_half_i_jphalf*U0_half_i_jphalf;
      F2_half_imhalf_j = U1_half_imhalf_j*U2_half_imhalf_j/U0_half_imhalf_j; G2_half_i_jmhalf = U2_half_i_jmhalf*U2_half_i_jmhalf/U0_half_i_jmhalf + 0.5*g*U0_half_i_jmhalf*U0_half_i_jmhalf;

      U0_np1[ij] = U0[ij] - (dt/dx)*(F0_half_iphalf_j - F0_half_imhalf_j) - (dt/dx)*(G0_half_i_jphalf - G0_half_i_jmhalf);
      U1_np1[ij] = U1[ij] - (dt/dx)*(F1_half_iphalf_j - F1_half_imhalf_j) - (dt/dx)*(G1_half_i_jphalf - G1_half_i_jmhalf);
      U2_np1[ij] = U2[ij] - (dt/dx)*(F2_half_iphalf_j - F2_half_imhalf_j) - (dt/dx)*(G2_half_i_jphalf - G2_half_i_jmhalf);
    }
  }

}

void print_data(float** hist,int iterations,int maxX,int maxY,float deltaR,int nB,int nT,float totalTime){
  float deltaX = deltaR;
  float deltaY = deltaR;
  int Nx = (int)(maxX/deltaX);
  int Ny = (int)(maxY/deltaY);
  int print_iterations = iterations/SAVE_ITERATION;
  //Add time to filename
  time_t t = time(NULL);
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
  fprintf(metaFile,"R step size: %lf\n",deltaR);
  fprintf(metaFile,"Maximum X: %d\n",maxX);
  fprintf(metaFile,"Maximum Y: %d\n",maxY);
  fprintf(metaFile,"Iterations: %d\n",iterations);
  fprintf(metaFile,"Number of blocks: %d\n",nB);
  fprintf(metaFile,"Number of threads: %d\n",nT);

  //Print R to binary
  fwrite(hist[0],sizeof(float)*Nx,1,xFile);
  fwrite(hist[1],sizeof(float)*Ny,1,yFile);
  //Print data to binary
  fwrite(hist[2],sizeof(float)*Nx*Ny*(iterations/SAVE_ITERATION),1,binFile);
}


int main(){

  int N = 500;
  int Nt = 5000;

  int count = 0;

  float dx = 0.01;
  float dt = 0.001;
  float g = 9.8;

  int maxX = N*dx;
  int maxY = N*dx;

  float x0 = -3;
  float y0 = -2;
  float sigma = 0.1;
  float A = 0.5;

  float* U0 = malloc(N*N*sizeof(float));
  float* U1 = malloc(N*N*sizeof(float));
  float* U2 = malloc(N*N*sizeof(float));
  float* U0_np1 = malloc(N*N*sizeof(float));
  float* U1_np1 = malloc(N*N*sizeof(float));
  float* U2_np1 = malloc(N*N*sizeof(float));
  float* temp0;
  float* temp1;
  float* temp2;
  float **hist = malloc(sizeof(float*)*3);
  float *H_hist = malloc(sizeof(float)*N*N*(Nt/SAVE_ITERATION));
  float *X_hist = malloc(sizeof(float)*N);
  float *Y_hist = malloc(sizeof(float)*N);

  initialise_fields(U0, U1, U2, N, dx, x0, y0, sigma, A);
  // output_height(U0, N);

  clock_t initTime = clock();
  printf("iteration started\n");
  for (int t=0; t < Nt; t++){
    evolve_fields(U0, U1, U2, U0_np1, U1_np1, U2_np1, N, dx, dt, g);
    temp0 = U0; temp1 = U1; temp2 = U2; // Swap pointers for next iteration
    U0 = U0_np1; U1 = U1_np1; U2 = U2_np1;
    U0_np1 = temp0; U1_np1 = temp1; U2_np1 = temp2;
    if (t % 10 == 0){
        //Save values of u and u_t
        //printf("iteration %d\n",i);
        for(int y=0;y<N;y++){
            for(int x=0;x<N;x++){
                H_hist[(t/SAVE_ITERATION)*N*N +N*y +x] = U0[N*y +x];
            }
        }
    }
  }
  for(int ix=0;ix<N;ix++)
      X_hist[ix] = ix*dx;
  for(int iy=0;iy<N;iy++)
      Y_hist[iy] = iy*dx;
  hist[0] =  X_hist;
  hist[1] =  Y_hist;
  hist[2] =  H_hist;

  clock_t finalTime = clock();
  float  totalTime = (float)(finalTime-initTime) / CLOCKS_PER_SEC;
  printf("iteration finished\n");
  print_data(hist,Nt,maxX,maxY,dx,1,1,totalTime);

  // output_height(U0, N);

}
