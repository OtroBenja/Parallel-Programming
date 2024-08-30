#include <stdio.h>
#include <stdlib.h>

__global__ void imprimir_gpu(){
    printf("Hello world from thread [%d,%d] of device", threadIdx.x, blockIdx.x);
}

int main(){
    printf("Hello world from host");
}