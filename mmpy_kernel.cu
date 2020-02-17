// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include <algorithm>
#include "utils.h"
#include "types.h"

using namespace std;

#define BLOCK_SIZE 32

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

    /*if(N == 256) {
        I0 = blockIdx.y*BLOCK_SIZE+ty;
        J0 = blockIdx.x*BLOCK_SIZE+tx;

        int nBLOCK_SIZE = N/BLOCK_SIZE;
        #pragma unroll 4
        for (unsigned int kk=0;kk<nBLOCK_SIZE;++kk) {
            As0[ty][tx] = A[I0*N + kk*BLOCK_SIZE+tx];
            Bs0[ty][tx] = B[(kk*BLOCK_SIZE+ty)*N + J0];

            __syncthreads();
            #pragma unroll 32
            for (unsigned int k=0;k<BLOCK_SIZE;++k) {
                Cij[0] += As0[ty][k] * Bs0[k][tx];
            }
            __syncthreads();
        }
        C[I0*N+J0] = Cij[0];
    }
    else */if (!(N&(N-1))) { // powers of 2
        int nBLOCK_SIZE = N/BLOCK_SIZE;
        #pragma unroll 4
        for (unsigned int kk=0;kk<nBLOCK_SIZE;++kk) {
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
    else { // non-powers of 2 (with boundary checks)
        int ty = threadIdx.y;
        int tx = threadIdx.x;
        int by0 = 2*blockIdx.y;
        int bx0 = 2*blockIdx.x;

        int I0 = by0*BLOCK_SIZE+ty;
        int I1 = (by0+1)*BLOCK_SIZE+ty;

        int J0 = bx0*BLOCK_SIZE+tx;
        int J1 = (bx0+1)*BLOCK_SIZE+tx;

        _DOUBLE_ Cij[4] = {0,0,0,0};

        int nBLOCK_SIZE = (N/BLOCK_SIZE) + 1;
        
        #pragma unroll 4
        for (unsigned int kk=0;kk<nBLOCK_SIZE;++kk) {
            As0[ty][tx] = (I0<N && (kk*BLOCK_SIZE+tx) < N) ? A[I0*N + kk*BLOCK_SIZE+tx] : 0;
            As0[ty+32][tx] = (I1<N && (kk*BLOCK_SIZE+tx) < N) ? A[I1*N + kk*BLOCK_SIZE+tx] : 0;

            Bs0[ty][tx] = ((kk*BLOCK_SIZE+ty) < N  && J0<N) ? B[(kk*BLOCK_SIZE+ty)*N + J0] : 0;
            Bs0[ty][tx+32] = ((kk*BLOCK_SIZE+ty) < N  && J1<N) ? B[(kk*BLOCK_SIZE+ty)*N + J1] : 0;

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
        if(I0<N && J0<N)
          C[I0*N+J0] = Cij[0];
        if(I0<N && J1<N)
          C[I0*N+J1] = Cij[1];
        if(I1<N && J0<N)
          C[I1*N+J0] = Cij[2];
        if(I1<N && J1<N)
          C[I1*N+J1] = Cij[3];
    }
}
