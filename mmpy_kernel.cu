// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include <algorithm>
#include "utils.h"
#include "types.h"


using namespace std;

#define BLOCK_SIZE 32

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
    __shared__ _DOUBLE_ As0[BLOCK_SIZE][BLOCK_SIZE], Bs0[BLOCK_SIZE][BLOCK_SIZE];//, As1[BLOCK_SIZE][BLOCK_SIZE], Bs1[BLOCK_SIZE][BLOCK_SIZE];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int by0 = blockIdx.y;
    int bx0 = blockIdx.x;

    int I0_0 = by0*BLOCK_SIZE+ty;
    int I0_8 = by0*BLOCK_SIZE+ty+8;
    int I0_16 = by0*BLOCK_SIZE+ty+16;
    int I0_24 = by0*BLOCK_SIZE+ty+24;

    int J0 = bx0*BLOCK_SIZE+tx;

    _DOUBLE_ Cij[4] = {0,0,0,0};

    int nBLOCK_SIZE = N/BLOCK_SIZE;
#pragma unroll 4
    for (int kk=0;kk<nBLOCK_SIZE;kk+=1) { // should be less than or equal to, right?
        As0[ty][tx] = A[I0_0*N + kk*BLOCK_SIZE+tx];
        As0[ty+8][tx] = A[I0_8*N + kk*BLOCK_SIZE+tx];
        As0[ty+16][tx] = A[I0_16*N + kk*BLOCK_SIZE+tx];
        As0[ty+24][tx] = A[I0_24*N + kk*BLOCK_SIZE+tx];

        Bs0[ty][tx] = B[(kk*BLOCK_SIZE+ty)*N + J0];
        Bs0[ty+8][tx] = B[(kk*BLOCK_SIZE+ty+8)*N + J0];
        Bs0[ty+16][tx] = B[(kk*BLOCK_SIZE+ty+16)*N + J0];
        Bs0[ty+24][tx] = B[(kk*BLOCK_SIZE+ty+24)*N + J0];

        __syncthreads();
#pragma unroll 32  
        for (int k=0;k<BLOCK_SIZE;++k) {
            Cij[0] += As0[ty][k] * Bs0[k][tx];
            Cij[1] += As0[ty+8][k] * Bs0[k][tx];
            Cij[2] += As0[ty+16][k] * Bs0[k][tx];
            Cij[3] += As0[ty+24][k] * Bs0[k][tx];
        }
        __syncthreads();
    }
    C[I0_0*N+J0] = Cij[0];
    C[I0_8*N+J0] = Cij[1];
    C[I0_16*N+J0] = Cij[2];
    C[I0_24*N+J0] = Cij[3];
}
