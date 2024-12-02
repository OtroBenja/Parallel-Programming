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

float* initialize_field(float p0,float x0, float y0,float q,float deltaX, float deltaY,int Nx,int Ny){
    float* h = (float*)malloc(sizeof(float)*Nx*Ny);

    for(int y=0;y<Ny;y++){
        for(int x=0;x<Nx;x++){
            h[y*Nx+x] = 1.0+ p0*powf(E,-(pow(x*deltaX-x0,2)+pow(y*deltaY-y0,2))/q);
        }
    }
    return h;
}

float** iteration(float* h,float *hu, float *hv, int Nx, int Ny,
                  float deltaX, float deltaY, float deltaT, int iterations,int Nsave){
    float g = 9.8;
    int save_iter = (iterations-1)/(Nsave+1) +1;
    int size = sizeof(float)*Nx*Ny;
    float  *h_i05 = malloc(sizeof(float)*(Nx-1)*Ny);
    float *hu_i05 = malloc(sizeof(float)*(Nx-1)*Ny);
    float *hv_i05 = malloc(sizeof(float)*(Nx-1)*Ny);
    float  *h_j05 = malloc(sizeof(float)*Nx*(Ny-1));
    float *hu_j05 = malloc(sizeof(float)*Nx*(Ny-1));
    float *hv_j05 = malloc(sizeof(float)*Nx*(Ny-1));
    float *H_hist = malloc(sizeof(float)*(Nx-2)*(Ny-2)*(Nsave+1));
    float *X_hist = malloc(sizeof(float)*(Nx-2));
    float *Y_hist = malloc(sizeof(float)*(Ny-2));
    float **hist = malloc(sizeof(float*)*3);
    int i_save = 0;
    save_iter;

    for(int i=0;i<iterations;i++){
        //Save values of h
        if(i==save_iter*(i_save)){
            for(int y=0;y<(Ny-2);y++){
                for(int x=0;x<(Nx-2);x++){
                    H_hist[(i_save)*(Ny-2)*(Nx-2) +(Nx-2)*y +x] = h[Nx*(y+1) +(x+1)];
                }
            }
            i_save++;
        }

        //Set borders for boundary condition
        for(int x=0;x<Nx;x++){
             h[x] =  h[Nx*2 +x];
            hu[x] = -hu[Nx*2 +x];
            hv[x] = 0;//hv[Nx*2 +x];
             h[Nx*(Ny-1)+x] =  h[Nx*(Ny-3)+x];
            hu[Nx*(Ny-1)+x] = -hu[Nx*(Ny-3)+x];
            hv[Nx*(Ny-1)+x] = 0;//hv[Nx*(Ny-3)+x];
        }
        for(int y=0;y<Ny;y++){
             h[Nx*y] =  h[Nx*y +2];
            hu[Nx*y] = 0;//hu[Nx*y +2];
            hv[Nx*y] = -hv[Nx*y +2];
             h[Nx*y+Nx-1] =  h[Nx*y+Nx-3];
            hu[Nx*y+Nx-1] = 0;//hu[Nx*y+Nx-3];
            hv[Nx*y+Nx-1] = -hv[Nx*y+Nx-3];
        }

        //calculate half step for h, h*u and h*v
        for(int y=0;y<Ny;y++){
            for(int x=0;x<Nx-1;x++){
                 h_i05[(Nx-1)*y+x] = 0.5*( h[Nx*y+x+1]+ h[Nx*y+x  ])+0.5*deltaT*(hu[Nx*y+x+1]-hu[Nx*y+x])/deltaX;
                hv_i05[(Nx-1)*y+x] = 0.5*(hv[Nx*y+x+1]+hv[Nx*y+x  ])
                             +0.5*deltaT*(hu[Nx*y+x+1]*hv[Nx*y+x+1]/h[Nx*y+x+1]
                                         -hu[Nx*y+x  ]*hv[Nx*y+x  ]/h[Nx*y+x  ])/deltaX;
                hu_i05[(Nx-1)*y+x] = 0.5*(hu[Nx*y+x+1]+hu[Nx*y+x  ])
                             +0.5*deltaT*(hu[Nx*y+x+1]*hu[Nx*y+x+1]/h[Nx*y+x+1] +0.5*g*h[Nx*y+x+1]*h[Nx*y+x+1]
                                         -hu[Nx*y+x  ]*hu[Nx*y+x  ]/h[Nx*y+x  ] -0.5*g*h[Nx*y+x  ]*h[Nx*y+x  ])/deltaX;
            }
        }

        for(int y=0;y<Ny-1;y++){
            for(int x=0;x<Nx;x++){
                 h_j05[Nx*y+x] = 0.5*( h[Nx*(y+1)+x]+ h[Nx* y   +x])+0.5*deltaT*(hv[Nx*(y+1)+x]-hv[Nx*y+x])/deltaY;
                hu_j05[Nx*y+x] = 0.5*(hu[Nx*(y+1)+x]+hu[Nx* y   +x])
                         +0.5*deltaT*(hu[Nx*(y+1)+x]*hv[Nx*(y+1)+x]/h[Nx*(y+1)+x]
                                     -hu[Nx* y   +x]*hv[Nx* y   +x]/h[Nx* y   +x])/deltaY;
                hv_j05[Nx*y+x] = 0.5*(hv[Nx*(y+1)+x]+hv[Nx* y   +x])
                         +0.5*deltaT*(hv[Nx*(y+1)+x]*hv[Nx*(y+1)+x]/h[Nx*(y+1)+x] +0.5*g*h[Nx*(y+1)+x]*h[Nx*(y+1)+x]
                                     -hv[Nx* y   +x]*hv[Nx* y   +x]/h[Nx* y   +x] -0.5*g*h[Nx* y   +x]*h[Nx* y   +x])/deltaY;
            }
        }

        //Calculate next step for h, h*u and h*v using the previous half step
        for(int y=1;y<Ny-1;y++){
            for(int x=1;x<Nx-1;x++){
                 h[Nx*y+x] += deltaT*(hu_i05[(Nx-1)*y+x  ]-hu_i05[(Nx-1)*y+x-1])/deltaX +deltaT*(hv_j05[Nx*y+x]-hv_j05[Nx*(y-1)+x])/deltaY;
                hu[Nx*y+x] += deltaT*(hu_j05[Nx* y   +x]*hv_j05[Nx* y   +x]/h_j05[Nx* y   +x]
                                     -hu_j05[Nx*(y-1)+x]*hv_j05[Nx*(y-1)+x]/h_j05[Nx*(y-1)+x])/deltaY
                             +deltaT*(hu_i05[(Nx-1)*y+x  ]*hu_i05[(Nx-1)*y+x  ]/h_i05[(Nx-1)*y+x  ] +0.5*g*h_i05[(Nx-1)*y+x  ]*h_i05[(Nx-1)*y+x  ]
                                     -hu_i05[(Nx-1)*y+x-1]*hu_i05[(Nx-1)*y+x-1]/h_i05[(Nx-1)*y+x-1] -0.5*g*h_i05[(Nx-1)*y+x-1]*h_i05[(Nx-1)*y+x-1])/deltaX;   
                hv[Nx*y+x] += deltaT*(hu_i05[(Nx-1)*y+x  ]*hv_i05[(Nx-1)*y+x  ]/h_i05[(Nx-1)*y+x  ]
                                     -hu_i05[(Nx-1)*y+x-1]*hv_i05[(Nx-1)*y+x-1]/h_i05[(Nx-1)*y+x-1])/deltaX
                             +deltaT*(hu_j05[Nx* y   +x]*hu_j05[Nx* y   +x]/h_j05[Nx* y   +x] +0.5*g*h_j05[Nx* y   +x]*h_j05[Nx* y   +x]
                                     -hu_j05[Nx*(y-1)+x]*hu_j05[Nx*(y-1)+x]/h_j05[Nx*(y-1)+x] -0.5*g*h_j05[Nx*(y-1)+x]*h_j05[Nx*(y-1)+x])/deltaY;
            }
        }
    }
    //Save last values of h
    for(int y=0;y<(Ny-2);y++){
        for(int x=0;x<(Nx-2);x++){
            H_hist[(Nsave)*(Ny-2)*(Nx-2) +(Nx-2)*y +x] = h[Nx*(y+1) +(x+1)];
        }
    }

    for(int ix=0;ix<(Nx-2);ix++)
        X_hist[ix] = ix*deltaX;
    for(int iy=0;iy<(Ny-2);iy++)
        Y_hist[iy] = iy*deltaY;
    hist[0] =  X_hist;
    hist[1] =  Y_hist;
    hist[2] =  H_hist;

    return hist;
}

void print_data(float** hist,int iterations,int Nsave,int Nx,int Ny,float deltaX,float deltaY,float deltaT,int nB,int nT,float totalTime){
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
    fprintf(metaFile,"Tme step size: %f\n",deltaT);
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

void main(int argc, char* argv[]){
    
    //Define initial conditions
    int fType = 0;
    float p0 = 0.4;
    float x0 = 1.5;
    float y0 = 2.0;
    float q = 0.1;
    
    //Define simulation limits
    int iterations = ITERATIONS;
    if((argc>1) && atoi(argv[1])) iterations = atoi(argv[1]);
    int Nx = 300;
    int Ny = 300;
    if((argc>2) && atoi(argv[2])) Nx = atoi(argv[2]);
    if((argc>3) && atoi(argv[3])) Ny = atoi(argv[3]);
    float deltaR = 0.01;
    float deltaX = deltaR;
    float deltaY = deltaR;
    float cfl = 50;
    if((argc>4) && atoi(argv[4])) cfl = atof(argv[4]);
    float deltaT=deltaR/cfl;
    int Nsave = 0;
    if((argc>5) && atoi(argv[5])) Nsave = atoi(argv[5]);

    //Allocate memory on host
    int size = sizeof(float)*Nx*Ny;
    float *hu  = (float*)malloc(size);
    float *hv  = (float*)malloc(size);
    float **hist;

    float* h;
    h = initialize_field(p0,x0,y0,q,deltaX,deltaY,Nx,Ny);
    //Set initial velocities to zero
    for(int y=0;y<Ny;y++){
        for(int x=0;x<Nx;x++){
            hu[Nx*y+x] = 0;
            hv[Nx*y+x] = 0;
        }
    }

    //Pass initial conditions to iteration
    int save_iter = iterations/(Nsave+1);
    clock_t initTime = clock();
    printf("iteration started\n");
    hist = iteration(h,hu,hv,Nx,Ny,deltaX,deltaY,deltaT,iterations,Nsave);
    clock_t finalTime = clock();
    float  totalTime = (float)(finalTime-initTime) / CLOCKS_PER_SEC;
    printf("iteration finished\n");

    //Print simulation history to a file
    printf("saving data...");
    print_data(hist,iterations,Nsave+1,Nx,Ny,deltaX,deltaY,deltaT,1,1,totalTime);
    printf("\tData saved to files\n");
    printf("All finished\n");

}


