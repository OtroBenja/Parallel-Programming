#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "shallow_waters.h"

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

void main(int argc, char* argv[]){
    
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
    float **hist;

    float* h;
    h = initialize_field(a0,x0,y0,q,deltaX,deltaY,Nx,Ny);
    //Set initial velocities to zero
    for(int y=0;y<Ny;y++){
        for(int x=0;x<Nx;x++){
            hu[Nx*y+x] = 0;
            hv[Nx*y+x] = 0;
        }
    }

    //Pass initial conditions to iteration
    printf("iteration started\n");
    clock_t initTime = clock();
    hist = iteration(h,hu,hv,Nx,Ny,deltaX,deltaY,deltaT,iterations,Nsave);
    clock_t finalTime = clock();
    float  totalTime = (float)(finalTime-initTime) / CLOCKS_PER_SEC;
    printf("iteration finished\n");

    //Print simulation history to a file
    printf("saving data...");
    char exec_type[] = "Sequential CPU";
    print_data(hist,exec_type,iterations,Nsave+1,Nx,Ny,deltaX,deltaY,deltaT,1,1,totalTime);
    printf("\tData saved to files\n");
    printf("All finished\n");

}


