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

double leftmost_D1(double *f,int idx,double deltaR){
    return (-25.0*f[idx] +48.0*f[idx+1] -36.0*f[idx+2] +16.0*f[idx+3] -3.0*f[idx+4])/(12.0*deltaR);
}
double leftmid_D1(double *f,int idx,double deltaR){
    return (-3.0*f[idx-1] -10.0*f[idx] +18.0*f[idx+1] -6.0*f[idx+2] +1.0*f[idx+3])/(12.0*deltaR);
}
double centered_D1(double *f,int idx,double deltaR){
    return (f[idx-2] -8.0*f[idx-1] +8.0*f[idx+1] -f[idx+2])/(12.0*deltaR);
}
double rightmid_D1(double *f,int idx,double deltaR){
    return (-1.0*f[idx-3] +6.0*f[idx-2] -18.0*f[idx-1] +10.0*f[idx] +3.0*f[idx+1])/(12.0*deltaR);
}
double rightmost_D1(double *f,int idx,double deltaR){
    return (3.0*f[idx-4] -16.0*f[idx-3] +36.0*f[idx-2] -48.0*f[idx-1] +25.0*f[idx])/(12.0*deltaR);
}

float* initialize_field(float p0,float x0, float y0,float q,float deltaR,int maxX,int maxY){
    float deltaX = deltaR;
    float deltaY = deltaR;
    int Nx = maxX/deltaR;
    int Ny = maxY/deltaR;
    float* h = malloc(sizeof(float)*Nx*Ny);

    for(int y=0;y<Ny;y++){
        for(int x=0;x<Nx;x++){
            h[y*Nx+x] = 1.0+ p0*powf(E,-(pow(x*deltaX-x0,2)+pow(y*deltaY-y0,2))/q);
        }
    }
    return h;
}

float** iteration(float* h,float deltaR,float maxX,float maxY,int iterations,int save_iteration){
    float deltaX = deltaR;
    float deltaY = deltaR;
    int Nx = maxX/deltaX;
    int Ny = maxY/deltaY;
    float g = 9.8;
    int size = sizeof(float)*Nx*Ny;
    float deltaT=deltaR/10.;
    //float *u  = malloc(sizeof(float)*Nx*Ny);
    //float *v  = malloc(sizeof(float)*Nx*Ny);
    float *hu = malloc(size);
    float *hv = malloc(size);
    float  *h_i05 = malloc(sizeof(float)*(Nx-1)*Ny);
    float *hu_i05 = malloc(sizeof(float)*(Nx-1)*Ny);
    float *hv_i05 = malloc(sizeof(float)*(Nx-1)*Ny);
    float  *h_j05 = malloc(sizeof(float)*Nx*(Ny-1));
    float *hu_j05 = malloc(sizeof(float)*Nx*(Ny-1));
    float *hv_j05 = malloc(sizeof(float)*Nx*(Ny-1));
    float *H_hist = malloc(sizeof(float)*(Nx-2)*(Ny-2)*(iterations/save_iteration));
    float *X_hist = malloc(sizeof(float)*(Nx-2));
    float *Y_hist = malloc(sizeof(float)*(Ny-2));
    float **hist = malloc(sizeof(float*)*3);
    int save_count = save_iteration;

    //Set initial velocities to zero
    for(int y=0;y<Ny;y++){
        for(int x=0;x<Nx;x++){
             //u[Nx*y+x] = 0;
             //v[Nx*y+x] = 0;
            hu[Nx*y+x] = 0;
            hv[Nx*y+x] = 0;
        }
    }

    for(int i=0;i<iterations;i++){
        //Save values of u and u_t
        if(save_count == save_iteration){
            //printf("iteration %d\n",i);
            for(int y=0;y<(Ny-2);y++){
                for(int x=0;x<(Nx-2);x++){
                    H_hist[(i/save_iteration)*(Ny-2)*(Nx-2) +(Nx-2)*y +x] = h[Nx*(y+1) +(x+1)];
                }
            }
            save_count=0;
        }
        save_count+=1;

        //Set borders for boundary condition
        for(int x=0;x<Nx;x++){
             h[x] =  h[Nx*2 +x];
             //u[x] = -u[Nx*2 +x];
             //v[x] = 0;//v[Nx*2 +x];
            hu[x] = -hu[Nx*2 +x];
            hv[x] = 0;//hv[Nx*2 +x];
             h[Nx*(Ny-1)+x] =  h[Nx*(Ny-3)+x];
             //u[Nx*(Ny-1)+x] = -u[Nx*(Ny-3)+x];
             //v[Nx*(Ny-1)+x] = 0;//v[Nx*(Ny-3)+x];
            hu[Nx*(Ny-1)+x] = -hu[Nx*(Ny-3)+x];
            hv[Nx*(Ny-1)+x] = 0;//hv[Nx*(Ny-3)+x];
        }
        for(int y=0;y<Ny;y++){
             h[Nx*y] =  h[Nx*y +2];
             //u[Nx*y] = 0;//u[Nx*y +2];
             //v[Nx*y] = -v[Nx*y +2];
            hu[Nx*y] = 0;//hu[Nx*y +2];
            hv[Nx*y] = -hv[Nx*y +2];
             h[Nx*y+Nx-1] =  h[Nx*y+Nx-3];
             //u[Nx*y+Nx-1] = 0;//u[Nx*y+Nx-3];
             //v[Nx*y+Nx-1] = -v[Nx*y+Nx-3];
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
                 //u[Nx*y+x] = hu[Nx*y+x]/h[Nx*y+x];
                 //v[Nx*y+x] = hv[Nx*y+x]/h[Nx*y+x];
            }
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

void main(int argc, char* argv[]){
    float* u;
    float** hist;
    float* r;
    float* phi;
    float* Phi;
    float* Pi;
    
    //Define initial conditions
    int fType = 0;
    float p0 = 0.5;
    float x0 = 3.0;
    float y0 = 2.0;
    float q = 0.1;
    
    //Define simulation limits
    int iterations = ITERATIONS;
    if((argc>1) && atoi(argv[1])) iterations = atoi(argv[1]);
    float maxX = 5;
    float maxY = 5;
    if((argc>2) && atoi(argv[2])) maxX = atoi(argv[2]);
    if((argc>3) && atoi(argv[3])) maxY = atoi(argv[3]);
    float nT = 1;
    if((argc>4) && atoi(argv[4])) nT = atoi(argv[4]);
    nT = 1;
    float nB = 1;
    if((argc>5) && atoi(argv[5])) nB = atoi(argv[5]);
    nB = 1;
    float deltaR = 0.01;

    u = initialize_field(p0,x0,y0,q,deltaR,maxX,maxY);
    printf("field initialized\n");

    //Pass initial conditions to iteration
    clock_t initTime = clock();
    printf("iteration started\n");
    hist = iteration(u,deltaR,maxX,maxY,iterations,SAVE_ITERATION);
    clock_t finalTime = clock();
    float  totalTime = (float)(finalTime-initTime) / CLOCKS_PER_SEC;
    printf("iteration finished\n");

    //Print simulation history to a file
    printf("saving data...");
    print_data(hist,iterations,maxX,maxY,deltaR,nT,nB,totalTime);
    printf("\tData saved to files\n");
    printf("All finished\n");

}


