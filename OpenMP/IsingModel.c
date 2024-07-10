#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

//#define N 16
//#define ITERATIONS 400000
//#define TEMPERATURE 2.0
#define J 1.0
#define B 0.0


void initialize_grid(int8_t* field, int N){
    int random_N;
    for(int i=0;i<N*N;i++){
        random_N = rand();
        if (random_N<RAND_MAX/2)    field[i]=-1;
        else                        field[i]=1;
    }
}

float get_total_energy(int8_t* field, int N){
    float total_energy = 0.;

    //Calculate energy of interaction between particles
    for(int i=0;i<N-1;i++){
        for(int j=0;j<N-1;j++){
            //Calculate interaction with down and right particles only
            total_energy+=-J*(field[N*i+j]*field[N*i+j+1]+field[N*i+j]*field[N*(i+1)+j]);
        }
    }
    //Calculate energy of remaining particles using cyclic frontier conditions
    //for the right side of the grid
    for(int i=0;i<N-1;i++)
        total_energy += -J*(field[N*i+N-1]*field[N*i]+field[N*i+N-1]*field[N*(i+1)+N-1]);
    //for the bottom part of the grid
    for(int j=0;j<N-1;j++)
        total_energy += -J*(field[N*(N-1)+j]*field[N*(N-1)+j+1]+field[N*(N-1)+j]*field[j]);
    //for the bottom-right corner of the grid
    total_energy += -J*(field[N*(N-1)+N-1]*field[N*(N-1)]+field[N*(N-1)+N-1]*field[N-1]);

    //Calculate energy from the external magnetic field
    for(int i=0;i<N*N;i++)
        total_energy += -B*field[i];

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

float calculate_magnetization(int8_t* field, int N) {
    int sum = 0;
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            sum += field[N*i+j];
        }
    }
    return fabs((float)sum/(N*N));
}

void fprint_data(float* E_hist,float* M_hist, int iter, int save_iter){
    FILE* values_file;
    values_file = fopen("IsingFile.csv","w");
    fprintf(values_file,"Iteration,Energy,Magnetization\n");
    for(int i=0;i<iter/save_iter;i++){
        fprintf(values_file,"%d,%.4f,%.4f\n",i*save_iter,E_hist[i],M_hist[i]);
    }
    fclose(values_file);
}

void fprint_field(int8_t* field_hist, int N, int iter, int save_iter){
    FILE* values_file;
    values_file = fopen("IsingField.csv","w");
    fprintf(values_file,"Field State\n");
    for(int i=0;i<iter/save_iter;i++){
        for(int n=0;n<N*N-1;n++){
            fprintf(values_file,"%d,",field_hist[i*N*N + n]);
        }
        fprintf(values_file,"%d\n",field_hist[(i+1)*N*N -1]);
    }
    fclose(values_file);
}

void main(int argc,char **argv){
    char *output;
    int iterations = (int)strtol(argv[1], &output, 10);
    int N = (int)strtol(argv[2], &output, 10);
    float temperature = (float)strtod(argv[3],NULL);
    int save_iteration = (int)strtol(argv[4], &output, 10);
    printf("Iterations: %d\n",iterations);
    printf("Side size: %d\n",N);
    printf("Temperature: %f\n",temperature);
    printf("Save every %d steps\n",save_iteration);

    int8_t* field = malloc(sizeof(int8_t)*N*N);
    int8_t* field_history = malloc(sizeof(int8_t)*N*N*(iterations/save_iteration));
    float* mean_energy_history = malloc(sizeof(float)*iterations/save_iteration);
    float* magnetization_history = malloc(sizeof(float)*iterations/save_iteration);
    float beta = 1/temperature;
    int save_count = save_iteration;

    //Generate initial random state
    initialize_grid(field,N);
    //Iterate the model
    for(int i=0;i<iterations;i++){
        //Select a random point and calculate ΔE
        int random_i = rand()*(N-1)/RAND_MAX;
        int random_j = rand()*(N-1)/RAND_MAX;
        float deltaE = -2*get_energy(field,random_i,random_j,N);

        //If ΔE <= 0 change, otherwise change based on a probability
        if (deltaE<=0 ||randf()<expf(-beta*deltaE)) field[N*random_i+random_j] *= -1;
        //Save magnetization and energy after a certain amount of iterations
        if(save_count == save_iteration){
            mean_energy_history[i/save_iteration] = get_total_energy(field,N)/(N*N);
            magnetization_history[i/save_iteration] = calculate_magnetization(field,N);
            //Save every value of the field for this iteration
            for (int n=0; n<N*N; n++) {
                    field_history[N*N*(i/save_iteration) + n] = field[n];
            }
            save_count=0;
        }
        save_count++;
    }

    //Save data to file
    fprint_data(mean_energy_history,magnetization_history,iterations,save_iteration);
    fprint_field(field_history,N,iterations,save_iteration);
    printf("Data saved to file, execution succesfull\n");

}