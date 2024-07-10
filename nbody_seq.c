#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <immintrin.h>

#define N 500
#define G 1.0
#define EPS 0.1
#define ITERATIONS 10000
#define DT 0.05

float magnitude(float* array){
    float array_sq;
    float mag;
    array_sq = array[0]*array[0]+array[1]*array[1]+array[2]*array[2];
    mag = sqrtf(array_sq);
    return mag;
}

void grav_acceleration(float* rx,float* ry,float* rz,float* m_array,float* ax,float* ay,float* az,float r_ij[3],float acceleration_ij[3]){
    for(int i=0;i<N;i++){
        //Initialize accelerations
        ax[i] = 0.;
        ay[i] = 0.;
        az[i] = 0.;
    }
    for(int i=0;i<N;i++){
        for(int j=0;j<i;j++){
            //Compute position differences
            r_ij[0] = rx[i] - rx[j];
            r_ij[1] = ry[i] - ry[j];
            r_ij[2] = rz[i] - rz[j];
            //Compute acceleration between a pair
            acceleration_ij[0] = -G*(r_ij[0])*(powf(magnitude(r_ij)+EPS,-3));
            acceleration_ij[1] = -G*(r_ij[1])*(powf(magnitude(r_ij)+EPS,-3));
            acceleration_ij[2] = -G*(r_ij[2])*(powf(magnitude(r_ij)+EPS,-3));
            //Add acceleration to each particle
            ax[i] += acceleration_ij[0]*m_array[j];
            ay[i] += acceleration_ij[1]*m_array[j];
            az[i] += acceleration_ij[2]*m_array[j];
            ax[j] += -acceleration_ij[0]*m_array[i];
            ay[j] += -acceleration_ij[1]*m_array[i];
            az[j] += -acceleration_ij[2]*m_array[i];
        }
    }
}

float total_energy(float* rx,float* ry,float* rz,float* m_array,float* vx,float* vy,float* vz){
    float energy = 0.;
    float r_ij[3];
    float speed2;
    //Add total gravitational potential energy
    for(int i=0;i<N;i++){
        for(int j=0;j<i;j++){
            //Compute position differences
            r_ij[0] = rx[i] - rx[j];
            r_ij[1] = ry[i] - ry[j];
            r_ij[2] = rz[i] - rz[j];
            //Add potential energy between a pair
            energy += -G*m_array[i]*m_array[j]/magnitude(r_ij);
        }
    }
    //Add total kinetic energy
    for(int n=0;n<N;n++){
        //Compute squared speed
        speed2  = vx[n]*vx[n];
        speed2 += vy[n]*vy[n];
        speed2 += vz[n]*vz[n];
        //Add kinetic energy of the particle
        energy += .5*m_array[n]*speed2;
    }
    return energy;
}

float* total_momentum(float* m_array,float* vx,float* vy,float* vz){
    float* total_mom=malloc(sizeof(float)*3);
    float total_mass=0.0;
    total_mom[0] = 0.;
    total_mom[1] = 0.;
    total_mom[2] = 0.;
    for(int n=0;n<N;n++){
        total_mass += m_array[n];
        total_mom[0] += vx[n]*m_array[n];
        total_mom[1] += vy[n]*m_array[n];
        total_mom[2] += vz[n]*m_array[n];
    }
    return total_mom;
}

int compare (const void * num1, const void * num2) {
   if(*(int*)num1 > *(int*)num2)
    return 1;
   else
    return -1;
}

float lagrange_radii(float* rx,float* ry,float* rz,float* m_array){
    float r_mc[3];
    float r[3];
    float* dist = malloc(sizeof(float)*N);
    float total_mass=0.0;
    float radii=0.0;
    r_mc[0] = 0.;
    r_mc[1] = 0.;
    r_mc[2] = 0.;
    for(int n=0;n<N;n++){
        total_mass += m_array[n];
        r_mc[0] += rx[n]*m_array[n];
        r_mc[1] += ry[n]*m_array[n];
        r_mc[2] += rz[n]*m_array[n];
    }
    r_mc[0] = r_mc[0]/total_mass;
    r_mc[1] = r_mc[1]/total_mass;
    r_mc[2] = r_mc[2]/total_mass;
    //Calculate distance from center of mass
    for(int n=0;n<N;n++){
        r[0] = r_mc[0] - rx[n];
        r[1] = r_mc[1] - ry[n];
        r[2] = r_mc[2] - rz[n];
        dist[n] = magnitude(r);
    }
    //Sort distances
    qsort(dist,N,sizeof(float),compare);
    //Pick middle one
    return dist[N/2];
}

float** nbody_leapfrog(float* rx,float* ry,float* rz,float* m,float* vx,float* vy,float* vz,int iterations,float dt, bool energy){
    float* r_hist = _mm_malloc(sizeof(float)*N*3*(iterations+1),32);
    float* v_hist = _mm_malloc(sizeof(float)*N*3*(iterations+1),32);
    float* a_hist = _mm_malloc(sizeof(float)*N*3*(iterations+1),32);
    float* lag_radii = _mm_malloc(sizeof(float)*(iterations),32);
    float** hist = malloc(sizeof(float*)*4);
    float* ax0 = _mm_malloc(sizeof(float)*N,32);
    float* ay0 = _mm_malloc(sizeof(float)*N,32);
    float* az0 = _mm_malloc(sizeof(float)*N,32);
    float* ax1 = _mm_malloc(sizeof(float)*N,32);
    float* ay1 = _mm_malloc(sizeof(float)*N,32);
    float* az1 = _mm_malloc(sizeof(float)*N,32);
    float* a_temp;
    float r_ij[3];
    float acceleration_ij[3];
    //Print initial energy and velocity
    if (energy){
        printf("Leapfrog initial energy: %f\n",total_energy(rx,ry,rz,m,vx,vy,vz));
        //float* total_vel = malloc(sizeof(float)*3);
        //total_vel = total_momentum(m,vx,vy,vz);
        //printf("Leapfrog x initial momentum: %f\n",total_vel[0]);
        //printf("Leapfrog y initial momentum: %f\n",total_vel[1]);
        //printf("Leapfrog z initial momentum: %f\n",total_vel[2]);
    }
    grav_acceleration(rx,ry,rz,m,ax0,ay0,az0,r_ij,acceleration_ij);
    for(int i=0;i<iterations;i++){
        lag_radii[i] = lagrange_radii(rx,ry,rz,m);
        for(int n=0;n<N;n++){
            //save acceleration in history
            a_hist[i*N*3+n*3  ] = ax0[n];
            a_hist[i*N*3+n*3+1] = ay0[n];
            a_hist[i*N*3+n*3+2] = az0[n];
            //save velocity in history
            v_hist[i*N*3+n*3  ] = vx[n];
            v_hist[i*N*3+n*3+1] = vy[n];
            v_hist[i*N*3+n*3+2] = vz[n];
            //save position in history
            r_hist[i*N*3+n*3  ] = rx[n];
            r_hist[i*N*3+n*3+1] = ry[n];
            r_hist[i*N*3+n*3+2] = rz[n];
        }
        for(int n=0;n<N;n++){
            //Update positions
            rx[n] = rx[n] + vx[n]*dt + ax0[n]*dt*dt/2.;
            ry[n] = ry[n] + vy[n]*dt + ay0[n]*dt*dt/2.;
            rz[n] = rz[n] + vz[n]*dt + az0[n]*dt*dt/2.;
        }
        //Get acceleration from new positions
        grav_acceleration(rx,ry,rz,m,ax1,ay1,az1,r_ij,acceleration_ij);
        for(int n=0;n<N;n++){
            //Update velocities
            vx[n] = vx[n] + (ax1[n] + ax0[n])*dt/2.;
            vy[n] = vy[n] + (ay1[n] + ay0[n])*dt/2.;
            vz[n] = vz[n] + (az1[n] + az0[n])*dt/2.;
        }
        //Rename accelerations for next iteration
        a_temp = ax0;
        ax0 = ax1;
        ax1 = a_temp;
        a_temp = ay0;
        ay0 = ay1;
        ay1 = a_temp;
        a_temp = az0;
        az0 = az1;
        az1 = a_temp;
    }
    //Print final energy and velocity
    if (energy){
        printf("Leapfrog final energy: %f\n",total_energy(rx,ry,rz,m,vx,vy,vz));
        //float* total_vel = malloc(sizeof(float)*3);
        //total_vel = total_momentum(m,vx,vy,vz);
        //printf("Leapfrog x final momentum: %f\n",total_vel[0]);
        //printf("Leapfrog y final momentum: %f\n",total_vel[1]);
        //printf("Leapfrog z final momentum: %f\n",total_vel[2]);
    }
    hist[0] = r_hist;
    hist[1] = v_hist;
    hist[2] = a_hist;
    hist[3] = lag_radii;
    return hist;
}

float randf(){
    float float_rand = (float)rand()/RAND_MAX;
    return float_rand;
}

void fill_values(float *initial_r,float *masses,float *initial_v){
    for(int n=0;n<N;n++){
        masses[n] = randf()*10.;

        initial_r[3*n] = randf()*8. -4.;
        initial_r[3*n+1] = randf()*8. -4.;
        initial_r[3*n+2] = randf()*8. -4.;
        //initial_r[3*n+2] = 0.;

        initial_v[3*n] = randf()*1. -.5;
        initial_v[3*n+1] = randf()*1. -.5;
        initial_v[3*n+2] = randf()*1. -.5;
        //initial_v[3*n+2] = 0.;
    }
}

void load_plummer(float* rx,float* ry,float* rz,float* masses,float* vx,float* vy,float* vz){
    int Np = N;
    FILE *fp;    
    fp = fopen("PlummerIC.txt","r");
    if (fp == NULL){
            exit(1);
        } 
    for (int i = 0; i < Np; i++){
            fscanf(fp, "%f %f %f %f %f %f %f", &masses[i], &rx[i], &ry[i], &rz[i], &vx[i], &vy[i], &vz[i]);
            if (feof(fp)){
                break;
            }
        }
    for (int i = 0; i < Np; i++){
        //printf("%e %e %e %e %e %e %e\n",masses[i], rx[i], ry[i], rz[i], vx[i], vy[i], vz[i]);
    }
}

void transform_velocity(float* mass,float* vx,float* vy,float*vz){
    float* total_vel = malloc(sizeof(float)*3);
    float total_mass = 0;
    for(int n=0;n<N;n++){
        total_mass += mass[n];
    }
    total_vel = total_momentum(mass,vx,vy,vz);
    total_vel[0] = total_vel[0]/total_mass;
    total_vel[1] = total_vel[1]/total_mass;
    total_vel[2] = total_vel[2]/total_mass;
    for(int n=0;n<N;n++){
        vx[n] += -total_vel[0];
        vy[n] += -total_vel[1];
        vz[n] += -total_vel[2];
    }
}

void main(){
    //Set initial positions and velocities
    float* initial_rx = _mm_malloc(sizeof(float)*N,32);
    float* initial_ry = _mm_malloc(sizeof(float)*N,32);
    float* initial_rz = _mm_malloc(sizeof(float)*N,32);
    float* masses = malloc(sizeof(float)*N);
    float* initial_vx = _mm_malloc(sizeof(float)*N,32);
    float* initial_vy = _mm_malloc(sizeof(float)*N,32);
    float* initial_vz = _mm_malloc(sizeof(float)*N,32);
    float** history=malloc(sizeof(float*)*3);
    float* r_history;
    float* v_history;
    float* a_history;
    float* lag_radii;
    int iterations=ITERATIONS;
    float dt=DT;
    FILE* r_data;
    FILE* v_data;
    FILE* a_data;
    FILE* lagrange_data;

    //Initialize random seed
    srand(time(NULL)); 

    //Fill initial values of position, mass, and velocity
    //fill_values(initial_r,masses,initial_v);

    //Load Plummer initial conditions
    load_plummer(initial_rx,initial_ry,initial_rz,masses,initial_vx,initial_vy,initial_vz);

    //Set total velocity to 0
    transform_velocity(masses,initial_vx,initial_vy,initial_vz);

    //Get initial time
    clock_t t_i = clock();
    //Run leapfrog simulation
    history = nbody_leapfrog(initial_rx,initial_ry,initial_rz,masses,initial_vx,initial_vy,initial_vz,iterations,dt,true);
    r_history = history[0];
    v_history = history[1];
    a_history = history[2];
    lag_radii = history[3];
    //Get final and total simulation time
    clock_t t_f = clock();
    double simulation_time = (double)(t_f - t_i) / CLOCKS_PER_SEC;
    //Save simulation data
    r_data = fopen("positions.dat","w");
    v_data = fopen("velocities.dat","w");
    a_data = fopen("accelerations.dat","w");
    lagrange_data = fopen("lagrangeR.dat","w");
    for(int i=0;i<iterations;i++){
        for(int n=0;n<N;n++){
            fprintf(r_data,"%f,",r_history[i*N*3+n*3]);
            fprintf(r_data,"%f,",r_history[i*N*3+n*3+1]);
            fprintf(r_data,"%f",r_history[i*N*3+n*3+2]);
            fprintf(v_data,"%f,",v_history[i*N*3+n*3]);
            fprintf(v_data,"%f,",v_history[i*N*3+n*3+1]);
            fprintf(v_data,"%f",v_history[i*N*3+n*3+2]);
            fprintf(a_data,"%f,",a_history[i*N*3+n*3]);
            fprintf(a_data,"%f,",a_history[i*N*3+n*3+1]);
            fprintf(a_data,"%f",a_history[i*N*3+n*3+2]);
            if(n<N-1){
                fprintf(r_data,",");
                fprintf(v_data,",");
                fprintf(a_data,",");
            }
        }
        fprintf(lagrange_data,"%f\n",lag_radii[i]);
        fprintf(r_data,"\n");
        fprintf(v_data,"\n");
        fprintf(a_data,"\n");
    }
    fclose(r_data);
    fclose(v_data);
    fclose(a_data);
    fclose(lagrange_data);
    printf("Total simulation time (sequential): %lf\n",simulation_time);
}