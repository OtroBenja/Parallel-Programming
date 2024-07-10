#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

//#define N 32
//#define ITERATIONS 400000
//#define TEMPERATURE 2.0
#define J 1.0
#define B 0.0


void initialize_grid(int8_t* field, int N){
    int random_N;
    for(int i=0; i<N*N; i++){
        random_N = rand();
        if (random_N<RAND_MAX/2)    field[i]=-1;
        else                        field[i]=1;
    }
}

float get_total_energy(int8_t* field, int N, int NP){
    float total_energy = 0.;

    //Calculate energy of interaction between particles
    #pragma omp parallel reduction(+:total_energy) num_threads(NP)
    {   
        float subtotal_energy = 0.;
        #pragma omp for
        for(int i=0; i<N-1; i++)
        {
            for(int j=0; j<N-1; j++){
                //Calculate interaction with down and right particles only
                subtotal_energy+=-J*(field[N*i+j]*field[N*i+j+1]+field[N*i+j]*field[N*(i+1)+j]);
            }
        }
        //Calculate energy of remaining particles using cyclic frontier conditions
        //for the bottom and right sides of the grid
        #pragma omp for
        for(int i=0; i<N-1; i++)
        {
            subtotal_energy += -J*(field[N*i+N-1]*field[N*i]+field[N*i+N-1]*field[N*(i+1)+N-1]); //Right
            subtotal_energy += -J*(field[N*(N-1)+i]*field[N*(N-1)+i+1]+field[N*(N-1)+i]*field[i]); //Bottom
        }
        //Calculate energy from the external magnetic field
        #pragma omp for
        for(int i=0; i<N*N; i++)
        {
            subtotal_energy += -B*field[i];
        }
        total_energy = subtotal_energy;
    }
    //for the bottom-right corner of the grid
    total_energy += -J*(field[N*(N-1)+N-1]*field[N*(N-1)]+field[N*(N-1)+N-1]*field[N-1]);

    return total_energy;
}

float get_energy(int8_t* field, int i, int j, int N){
    float energy = 0.;
    int8_t left;
    int8_t right;
    int8_t up;
    int8_t down;

    //Calculate energy of a particle with cyclic conditions
    if(j==0) left=field[N*i+N-1]; else left=field[N*i+j-1];
    if(j==N-1) right=field[N*i+0]; else right=field[N*i+j+1];
    if(i==0) up=field[N*(N-1)+j]; else up=field[N*(i-1)+j];
    if(i==N-1) down=field[N*0+j]; else down=field[N*(i+1)+j];
    energy = -J*field[N*i+j]*(left+right+up+down);

    //Calculate energy from the external magnetic field
    energy += -B*field[N*i+j];

    return energy;
}

float randf(){
    float float_rand = (float)rand()/RAND_MAX;
    return float_rand;
}

float calculate_magnetization(int8_t* field, int N, int NP) {
    int sum = 0;
    #pragma omp parallel for reduction(+:sum) num_threads(NP)
    for (int i=0; i<N; i++) 
    {
        for (int j=0; j<N; j++){
            sum += field[N*i+j];
        }
    }
    return fabs((float) sum/(N*N));
}

void print_data(float* E_hist,float* M_hist, int iter,int NP){
    FILE* values_file;
    char filename[50];
    sprintf(filename, "IsingFileParallel%dP.csv", NP);
    values_file = fopen(filename, "w");
    fprintf(values_file, "Iteration,Energy,Magnetization\n");
    for(int i=0; i<iter; i++){
        fprintf(values_file, "%d,%.4f,%.4f\n", i, E_hist[i], M_hist[i]);
    }
    fclose(values_file);
}


void main(int argc, char **argv){
    char *output;
    int iterations = (int)strtol(argv[1], &output, 10);
    int N = (int)strtol(argv[2], &output, 10);
    float temperature = (float)strtod(argv[3], NULL);
    int NP = (int)strtol(argv[4], &output, 10);
    printf("Iterations: %d\n", iterations);
    printf("Side size: %d\n", N);
    printf("Temperature: %f\n", temperature);
    printf("Number of threads: %d\n", NP);

    int8_t* field = malloc(sizeof(int8_t)*N*N);
    float* mean_energy_history = malloc(sizeof(float)*iterations-1);
    float* magnetization_history = malloc(sizeof(float)*iterations-1);
    int* random_i = malloc(sizeof(int)*NP);
    int* random_j = malloc(sizeof(int)*NP);
    float* probability = malloc(sizeof(float)*NP);
    float beta = 1/temperature;

    //Generate initial random state
    initialize_grid(field, N);

    //Iterate the model
    for(int i=0; i<iterations; i++){
        //Using checkerboard method, apply the algorithm
        //over even numbered spaces of the lattice
        #pragma omp parallel for num_threads(NP) 
        for(int s=0;s<N*N;s+=2)
        {
        //Calculate ΔE
        int s_i = s/N;
        int s_j = s -N*s_i;
        float deltaE = -2*get_energy(field, s_i, s_j, N);
        //If ΔE <= 0 change, otherwise change based on a probability
        if (deltaE<=0 ||randf()<expf(-beta*deltaE)) field[N*s_i+s_j] *= -1;
        }
        //Using checkerboard method, apply the algorithm
        //over odd numbered spaces of the lattice
        #pragma omp parallel for num_threads(NP)
        for(int s=1; s<N*N; s+=2)
        {
        //Calculate ΔE
        int s_i = s/N;
        int s_j = s -N*s_i;
        float deltaE = -2*get_energy(field, s_i, s_j, N);
        //If ΔE <= 0 change, otherwise change based on a probability
        if (deltaE<=0 ||randf()<expf(-beta*deltaE)) field[N*s_i+s_j] *= -1;
        }
        
        //Save magnetization and energy of the iteration
        mean_energy_history[i] = get_total_energy(field, N, NP)/(N*N);
        magnetization_history[i] = calculate_magnetization(field, N, NP);
    }

    printf("Metropolis-Hastings simulation completed\n");

    //Save data to file
    print_data(mean_energy_history, magnetization_history, iterations, NP);
    printf("Data saved to file, execution succesfull\n");
}