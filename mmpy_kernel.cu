// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include <algorithm>
#include "utils.h"
#include "types.h"

using namespace std;

#define BLOCK_SIZE 32

#if 0
__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
    __shared__ _DOUBLE_ As0[BLOCK_SIZE][BLOCK_SIZE], Bs0[BLOCK_SIZE][BLOCK_SIZE];//, As1[BLOCK_SIZE][BLOCK_SIZE], Bs1[BLOCK_SIZE][BLOCK_SIZE];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int by0 = blockIdx.y;
    int bx0 = blockIdx.x;

    int I0_0 = by0*BLOCK_SIZE+ty;
    int I0_4 = by0*BLOCK_SIZE+ty+4;
    int I0_8 = by0*BLOCK_SIZE+ty+8;
    int I0_12 = by0*BLOCK_SIZE+ty+12;
    int I0_16 = by0*BLOCK_SIZE+ty+16;
    int I0_20 = by0*BLOCK_SIZE+ty+20;
    int I0_24 = by0*BLOCK_SIZE+ty+24;
    int I0_28 = by0*BLOCK_SIZE+ty+28;

    int J0 = bx0*BLOCK_SIZE+tx;

    _DOUBLE_ Cij[8] = {0,0,0,0,0,0,0,0};

    int nBLOCK_SIZE = N/BLOCK_SIZE;
#pragma unroll 4
    for (unsigned int kk=0;kk<nBLOCK_SIZE;kk+=1) { // should be less than or equal to, right?
        As0[ty][tx] = A[I0_0*N + kk*BLOCK_SIZE+tx];
        As0[ty+4][tx] = A[I0_4*N + kk*BLOCK_SIZE+tx];
        As0[ty+8][tx] = A[I0_8*N + kk*BLOCK_SIZE+tx];
        As0[ty+12][tx] = A[I0_12*N + kk*BLOCK_SIZE+tx];
        As0[ty+16][tx] = A[I0_16*N + kk*BLOCK_SIZE+tx];
        As0[ty+20][tx] = A[I0_20*N + kk*BLOCK_SIZE+tx];
        As0[ty+24][tx] = A[I0_24*N + kk*BLOCK_SIZE+tx];
        As0[ty+28][tx] = A[I0_28*N + kk*BLOCK_SIZE+tx];


        Bs0[ty][tx] = B[(kk*BLOCK_SIZE+ty)*N + J0];
        Bs0[ty+4][tx] = B[(kk*BLOCK_SIZE+ty+4)*N + J0];
        Bs0[ty+8][tx] = B[(kk*BLOCK_SIZE+ty+8)*N + J0];
        Bs0[ty+12][tx] = B[(kk*BLOCK_SIZE+ty+12)*N + J0];
        Bs0[ty+16][tx] = B[(kk*BLOCK_SIZE+ty+16)*N + J0];
        Bs0[ty+20][tx] = B[(kk*BLOCK_SIZE+ty+20)*N + J0];
        Bs0[ty+24][tx] = B[(kk*BLOCK_SIZE+ty+24)*N + J0];
        Bs0[ty+28][tx] = B[(kk*BLOCK_SIZE+ty+28)*N + J0];

        __syncthreads();
#pragma unroll 32  
        for (unsigned int k=0;k<BLOCK_SIZE;++k) {
            Cij[0] += As0[ty][k] * Bs0[k][tx];
            Cij[1] += As0[ty+4][k] * Bs0[k][tx];
            Cij[2] += As0[ty+8][k] * Bs0[k][tx];
            Cij[3] += As0[ty+12][k] * Bs0[k][tx];
            Cij[4] += As0[ty+16][k] * Bs0[k][tx];
            Cij[5] += As0[ty+20][k] * Bs0[k][tx];
            Cij[6] += As0[ty+24][k] * Bs0[k][tx];
            Cij[7] += As0[ty+28][k] * Bs0[k][tx];
        }
        __syncthreads();
    }
    C[I0_0*N+J0] = Cij[0];
    C[I0_4*N+J0] = Cij[1];
    C[I0_8*N+J0] = Cij[2];
    C[I0_12*N+J0] = Cij[3];
    C[I0_16*N+J0] = Cij[4];
    C[I0_20*N+J0] = Cij[5];
    C[I0_24*N+J0] = Cij[6];
    C[I0_28*N+J0] = Cij[7];
}
#else
__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
    __shared__ _DOUBLE_ As0[2*BLOCK_SIZE][BLOCK_SIZE], Bs0[BLOCK_SIZE][2*BLOCK_SIZE];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int by0 = 2*blockIdx.y;
    int bx0 = 2*blockIdx.x;

    int I0 = by0*BLOCK_SIZE+ty;
    int I1 = (by0+1)*BLOCK_SIZE+ty;

    int J0 = bx0*BLOCK_SIZE+tx;
    int J1 = (bx0+1)*BLOCK_SIZE+tx;

    _DOUBLE_ Cij[4] = {0,0,0,0};

    int nBLOCK_SIZE = N/BLOCK_SIZE;
#pragma unroll 4
    for (unsigned int kk=0;kk<nBLOCK_SIZE;++kk) { // should be less than or equal to, right?
        As0[ty][tx] = A[I0*N + kk*BLOCK_SIZE+tx];
        As0[ty+32][tx] = A[I1*N + kk*BLOCK_SIZE+tx];

        Bs0[ty][tx] = B[(kk*BLOCK_SIZE+ty)*N + J0];
        Bs0[ty][tx+32] = B[(kk*BLOCK_SIZE+ty)*N + J1];

        __syncthreads();
#pragma unroll 32  
        for (unsigned int k=0;k<BLOCK_SIZE;++k) {
            Cij[0] += As0[ty][k] * Bs0[k][tx];
            Cij[1] += As0[ty][k] * Bs0[k][tx+32];
            Cij[2] += As0[ty+32][k] * Bs0[k][tx];
            Cij[3] += As0[ty+32][k] * Bs0[k][tx+32];
        }
        __syncthreads();
    }
    C[I0*N+J0] = Cij[0];
    C[I0*N+J1] = Cij[1];
    C[I1*N+J0] = Cij[2];
    C[I1*N+J1] = Cij[3];
}
#endif