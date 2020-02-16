// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include <algorithm>
#include "utils.h"
#include "types.h"

#define BLOCK_SIZE 32

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
    __shared__ _DOUBLE_ As0[BLOCK_SIZE][BLOCK_SIZE], Bs0[BLOCK_SIZE][BLOCK_SIZE], As1[BLOCK_SIZE][BLOCK_SIZE], Bs1[BLOCK_SIZE][BLOCK_SIZE];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int by0 = blockIdx.y;
    int bx0 = blockIdx.x;
    int by1 = blockIdx.y+1;
    int bx1 = blockIdx.x+1;

    int I0 = by0*BLOCK_SIZE+ty;
    int J0 = bx0*BLOCK_SIZE+tx;
    int I1 = by1*BLOCK_SIZE+ty;
    int J1 = bx1*BLOCK_SIZE+tx;
    
    _DOUBLE_ Cij0 = 0;

    int nBLOCK_SIZE = N/BLOCK_SIZE;
#pragma unroll 4
    for (int kk=0;kk<nBLOCK_SIZE;kk+=2) { // should be less than or equal to, right?
        As0[ty][tx] = A[I0*N + kk*BLOCK_SIZE+tx];
        Bs0[ty][tx] = B[(kk*BLOCK_SIZE+ty)*N + J0];
        As1[ty][tx] = A[I1*N + kk*BLOCK_SIZE+tx];
        Bs1[ty][tx] = B[(kk*BLOCK_SIZE+ty)*N + J1];
        
        __syncthreads();
#pragma unroll 32   
        for (int k=0;k<BLOCK_SIZE;++k) {
            Cij0 += As0[ty][k] * Bs0[k][tx] + As1[ty][k] * Bs1[k][tx];
        }
        __syncthreads();
    }
    C[I0*N+J0] = Cij0;
}
