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


double* initialize_field(double p0,double r0,double d,double q,double deltaR,int maxR){
    int nR = maxR/deltaR;
    double* u = malloc(sizeof(double)*nR);

    //Calculate initial u
    for(int i=0;i<nR;i++){
        u[i] = p0*pow(i*deltaR,3)*pow(E,-pow((i*deltaR-r0)/d,q));
    }
    return u;
}

double** iteration(double* u,double deltaR,int maxR,int iterations,int save_iteration){
    int nR = maxR/deltaR;
    double deltaT=deltaR/5.;
    double* u_t = malloc(sizeof(double)*nR);
    double* u_tt = malloc(sizeof(double)*nR);
    double*  UHistory = malloc(sizeof(double)*(nR/SAVE_RES)*(iterations/save_iteration));
    double* UtHistory = malloc(sizeof(double)*(nR/SAVE_RES)*(iterations/save_iteration));
    double*  RHistory = malloc(sizeof(double)*(nR/SAVE_RES));
    double** hist = malloc(sizeof(double*)*3);
    int save_count = save_iteration;

    //Set initial u_t and u_tt
    for(int i=0;i<nR;i++){
         u_t[i] = 0;
    }

    for(int i=0;i<iterations;i++){
        //Save values of u and u_t
        if(save_count == save_iteration){
            printf("iteration %d\n",i);
            for(int ir=0;ir<(nR/SAVE_RES);ir++){
                 UHistory[(i/save_iteration)*(nR/SAVE_RES)+(ir)] = u[ir*SAVE_RES];
                UtHistory[(i/save_iteration)*(nR/SAVE_RES)+(ir)] = u_t[ir*SAVE_RES];
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
    double r0 = 20.;
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

    u = initialize_field(p0,r0,d,q,deltaR,maxR);

    //Pass initial conditions to iteration
    clock_t initTime = clock();
    hist = iteration(u,deltaR,maxR,iterations,SAVE_ITERATION);
    clock_t finalTime = clock();
    double  totalTime = (double)(finalTime-initTime) / CLOCKS_PER_SEC;

    //Print simulation history to a file
    int printR = (maxR/deltaR)/SAVE_RES;
    print_data(hist,iterations,maxR,deltaR,nT,nB,totalTime);

}


