#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define num_entries 10000000

int x[num_entries];
int y[num_entries];
int z[num_entries];

int main() {

    double end_time, start_time, time_spent;

    #pragma omp parallel for
    for (int i = 0; i < num_entries; i++) {
        x[i] = i;
        y[i] = num_entries - i;
    }

    start_time = omp_get_wtime();

    for  (int i = 0; i < num_entries; i++) {
        z[i] = x[i] + y[i];
    }

    end_time = omp_get_wtime();
    time_spent = end_time - start_time;
    printf("Time in serial sum: %lf\n",time_spent);

    start_time = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < num_entries; i++) {
        z[i] = x[i] + y[i];
    }

    end_time = omp_get_wtime();
    time_spent = end_time - start_time;
    printf("Time in parallel sum: %lf\n",time_spent);

    #pragma omp parallel for
    for (int i = 0; i < num_entries; i++)
        if (z[i] - num_entries)
            printf("error at position %d",i);
}