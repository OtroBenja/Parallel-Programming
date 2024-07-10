#include <stdio.h>
#include <omp.h>

int main(){
    #pragma omp parallel// num_threads(4)
    {
        int i = omp_get_thread_num();
        int n = omp_get_num_threads();
        printf("Hello world from thread %d of %d.\n",i,n);
    }
} 