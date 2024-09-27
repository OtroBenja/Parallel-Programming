#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define SAVE_RES 500
#define SAVE_ITERATION 1000
#define ITERATIONS 10000
#define PI 3.141592653
#define E  2.718281828
#define C 1.0


double* initialize_field(double p0,double x0, double y0,double d,double q,double deltaR,int maxR){
    int Nx = maxR/deltaR;
    int Ny = maxR/deltaR;
    double* h = malloc(sizeof(double)*Nx*Ny);

    //Calculate initial u
    //Set initial values of h
    for(int y=0;y<Ny;y++){
        for(int x=0;x<Nx;x++){
            h[y*Nx+x] = 0;
        }
    }
    return h;
}

double** iteration(double* u,double deltaR,int maxR,int iterations,int save_iteration){
    int Nx = maxR/deltaR;
    int Ny = maxR/deltaR;
    int size = sizeof(double)*Nx*Ny;
    double deltaT=deltaR/5.;
    double* vx = malloc(size);
    double* vy = malloc(size);
    double* h_vx = malloc(size);
    double* h_vy = malloc(size);
    double*  H_hist = malloc((iterations/save_iteration)*size/(SAVE_RES*SAVE_RES));
    double*  R_hist = malloc(size/(SAVE_RES*SAVE_RES));
    double** hist = malloc(sizeof(double*)*3);
    int save_count = save_iteration;

    //Set initial velocities
    for(int y=0;y<Ny;y++){
        for(int x=0;x<Nx;x++){
              vx[Nx*y+x] = 0;
              vy[Nx*y+x] = 0;
            h_vx[Nx*y+x] = 0;
            h_vy[Nx*y+x] = 0;
        }
    }

    for(int i=0;i<iterations;i++){
        //Save values of u and u_t
        if(save_count == save_iteration){
            printf("iteration %d\n",i);
            for(int y=0;y<(Ny/SAVE_RES);y++){
                for(int x=0;x<(Nx/SAVE_RES);x++){
                    h_hist[(i/save_iteration)*(Nx/SAVE_RES)+x] = h[x*SAVE_RES];
                }
            }
            save_count=0;
        }
        save_count+=1;

        //calculate u_tt = c^2 *u_xx
        u_tt[0] = C*C*(2*u[0] -5*u[1] +4*u[2] -1*u[3])/(deltaR*deltaR);
        u_tt[nR-1] = C*C*(2*u[nR-1] -5*u[nR-2] +4*u[nR-3] -1*u[nR-4])/(deltaR*deltaR);
        for(int i=1;i<nR-1;i++){
            u_tt[i] = C*C*(u[i+1]-2*u[i] +u[i-1])/(deltaR*deltaR);
        }

        //Advance u_t and u
        for(int ir=0;ir<nR;ir++){
            u_t[ir] += u_tt[ir]*deltaT;
            u[ir] += u_t[ir]*deltaT;
        }
    }
        
    for(int ir=0;ir<(nR/SAVE_RES);ir++){
        RHistory[ir] = ir*SAVE_RES*deltaR;
    }
    hist[0] =  RHistory;
    hist[1] =  UHistory;
    hist[2] = UtHistory;
    return hist;
}

void print_data(double** hist,int iterations,int maxR,double deltaR,int nB,int nT,double totalTime){
    int print_iterations = iterations/SAVE_ITERATION;
    int printR = (maxR/deltaR)/SAVE_RES;
    //Add time to filename
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char fileName[50];
    snprintf(fileName, sizeof(fileName), "Output_%02d%02d%02d.dat", tm.tm_hour, tm.tm_min, tm.tm_sec);
    FILE* data = fopen(fileName,"w");

    //Print all parameters
    fprintf(data,"Execution type: Sequential CPU\n");
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
    //Print u_t
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

void main(int argc, char* argv[]){
    double* u;
    double** hist;
    double* r;
    double* phi;
    double* Phi;
    double* Pi;
    
    //Define initial conditions
    int fType = 0;
    double p0 = 0.001;
    double x0 = 20.;
    double y0 = 20.;
    double d = 3.;
    double q = 2.;
    
    //Define simulation limits
    int iterations = ITERATIONS;
    if((argc>1) && atoi(argv[1])) iterations = atoi(argv[1]);
    int maxR = 80;
    if((argc>2) && atoi(argv[2])) maxR = atoi(argv[2]);
    double nT = 1;
    if((argc>3) && atoi(argv[3])) nT = atoi(argv[3]);
    nT = 1;
    double nB = 1;
    if((argc>4) && atoi(argv[4])) nB = atoi(argv[4]);
    nB = 1;
    double deltaR = 0.01;

    u = initialize_field(p0,x0,y0,d,q,deltaR,maxR);

    //Pass initial conditions to iteration
    clock_t initTime = clock();
    hist = iteration(u,deltaR,maxR,iterations,SAVE_ITERATION);
    clock_t finalTime = clock();
    double  totalTime = (double)(finalTime-initTime) / CLOCKS_PER_SEC;

    //Print simulation history to a file
    int printR = (maxR/deltaR)/SAVE_RES;
    print_data(hist,iterations,maxR,deltaR,nT,nB,totalTime);

}


