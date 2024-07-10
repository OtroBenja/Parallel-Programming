#include <stdio.h>
//Include a condition in case open mp is not installed
#if defined(_OPENMP)
#include <omp.h>
#else 
int omp_get_num_threads(){return 1;}
int omp_get_thread_num(){return 1;}
#endif

int main(){
    //Share for cycles between threads
    int n = omp_get_max_threads();
    #pragma omp parallel for
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
        int t = omp_get_thread_num();
        printf("Hello world from thread %d of %d in cycle i,j = %d,%d\n",t,n,i,j);
        }
    }
} 