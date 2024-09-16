#include <stdio.h>
#include <stdlib.h>
//#include "hip/hip_runtime.h"
#include "hcc.h"

__global__ void imprimir_gpu(){
    printf("Hello world from thread [%d,%d] of device", threadIdx.x, blockIdx.x);
}

int main(){
    printf("Hello world from host");
    hipLaunchKernel(1,8,0,0,
        imprimir_gpu);
}