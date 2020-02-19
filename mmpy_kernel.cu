// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include <algorithm>
#include "utils.h"
#include "types.h"

using namespace std;

#define BLOCK_SIZE 16

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
    __shared__ _DOUBLE_ As0[4*BLOCK_SIZE][BLOCK_SIZE], Bs0[BLOCK_SIZE][4*BLOCK_SIZE];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int by0 = 4*blockIdx.y;
    int bx0 = 4*blockIdx.x;

    int I0 = by0*BLOCK_SIZE+ty;
    int I1 = (by0+1)*BLOCK_SIZE+ty;
    int I2 = (by0+2)*BLOCK_SIZE+ty;
    int I3 = (by0+3)*BLOCK_SIZE+ty;

    int J0 = bx0*BLOCK_SIZE+tx;
    int J1 = (bx0+1)*BLOCK_SIZE+tx;
    int J2 = (bx0+2)*BLOCK_SIZE+tx;
    int J3 = (bx0+3)*BLOCK_SIZE+tx;

    _DOUBLE_ Cij[16] = {0};

    if (!(N&(N-1))) { // powers of 2
        int nBLOCK_SIZE = N/BLOCK_SIZE;
        #pragma unroll
        for (unsigned int kk=0;kk<nBLOCK_SIZE;++kk) {
            As0[ty][tx] = A[I0*N + kk*BLOCK_SIZE+tx];
            As0[ty+BLOCK_SIZE][tx] = A[I1*N + kk*BLOCK_SIZE+tx];
            As0[ty+2*BLOCK_SIZE][tx] = A[I2*N + kk*BLOCK_SIZE+tx];
            As0[ty+3*BLOCK_SIZE][tx] = A[I3*N + kk*BLOCK_SIZE+tx];

            Bs0[ty][tx] = B[(kk*BLOCK_SIZE+ty)*N + J0];
            Bs0[ty][tx+BLOCK_SIZE] = B[(kk*BLOCK_SIZE+ty)*N + J1];
            Bs0[ty][tx+2*BLOCK_SIZE] = B[(kk*BLOCK_SIZE+ty)*N + J2];
            Bs0[ty][tx+3*BLOCK_SIZE] = B[(kk*BLOCK_SIZE+ty)*N + J3];

            __syncthreads();
            #pragma unroll
            for (unsigned int k=0;k<BLOCK_SIZE;++k) {
                Cij[0] += As0[ty][k] * Bs0[k][tx];
                Cij[1] += As0[ty][k] * Bs0[k][tx+BLOCK_SIZE];
                Cij[2] += As0[ty][k] * Bs0[k][tx+2*BLOCK_SIZE];
                Cij[3] += As0[ty][k] * Bs0[k][tx+3*BLOCK_SIZE];
                Cij[4] += As0[ty+BLOCK_SIZE][k] * Bs0[k][tx];
                Cij[5] += As0[ty+BLOCK_SIZE][k] * Bs0[k][tx+BLOCK_SIZE];
                Cij[6] += As0[ty+BLOCK_SIZE][k] * Bs0[k][tx+2*BLOCK_SIZE];
                Cij[7] += As0[ty+BLOCK_SIZE][k] * Bs0[k][tx+3*BLOCK_SIZE];
                Cij[8] += As0[ty+2*BLOCK_SIZE][k] * Bs0[k][tx];
                Cij[9] += As0[ty+2*BLOCK_SIZE][k] * Bs0[k][tx+BLOCK_SIZE];
                Cij[10] += As0[ty+2*BLOCK_SIZE][k] * Bs0[k][tx+2*BLOCK_SIZE];
                Cij[11] += As0[ty+2*BLOCK_SIZE][k] * Bs0[k][tx+3*BLOCK_SIZE];
                Cij[12] += As0[ty+3*BLOCK_SIZE][k] * Bs0[k][tx];
                Cij[13] += As0[ty+3*BLOCK_SIZE][k] * Bs0[k][tx+BLOCK_SIZE];
                Cij[14] += As0[ty+3*BLOCK_SIZE][k] * Bs0[k][tx+2*BLOCK_SIZE];
                Cij[15] += As0[ty+3*BLOCK_SIZE][k] * Bs0[k][tx+3*BLOCK_SIZE];
            }
            __syncthreads();
        }
        C[I0*N+J0] = Cij[0];
        C[I0*N+J1] = Cij[1];
        C[I0*N+J2] = Cij[2];
        C[I0*N+J3] = Cij[3];
        C[I1*N+J0] = Cij[4];
        C[I1*N+J1] = Cij[5];
        C[I1*N+J2] = Cij[6];
        C[I1*N+J3] = Cij[7];
        C[I2*N+J0] = Cij[8];
        C[I2*N+J1] = Cij[9];
        C[I2*N+J2] = Cij[10];
        C[I2*N+J3] = Cij[11];
        C[I3*N+J0] = Cij[12];
        C[I3*N+J1] = Cij[13];
        C[I3*N+J2] = Cij[14];
        C[I3*N+J3] = Cij[15];
    }
    else { // non-powers of 2 (with boundary checks)
        int nBLOCK_SIZE = (N/BLOCK_SIZE) + 1;
        #pragma unroll
        for (unsigned int kk=0;kk<nBLOCK_SIZE;++kk) {
            As0[ty][tx] = (I0<N && (kk*BLOCK_SIZE+tx) < N) ? A[I0*N + kk*BLOCK_SIZE+tx] : 0;
            As0[ty+BLOCK_SIZE][tx] = (I1<N && (kk*BLOCK_SIZE+tx) < N) ? A[I1*N + kk*BLOCK_SIZE+tx] : 0;
            As0[ty+2*BLOCK_SIZE][tx] = (I2<N && (kk*BLOCK_SIZE+tx) < N) ? A[I2*N + kk*BLOCK_SIZE+tx] : 0;
            As0[ty+3*BLOCK_SIZE][tx] = (I3<N && (kk*BLOCK_SIZE+tx) < N) ? A[I3*N + kk*BLOCK_SIZE+tx] : 0;

            Bs0[ty][tx] = ((kk*BLOCK_SIZE+ty) < N  && J0<N) ? B[(kk*BLOCK_SIZE+ty)*N + J0] : 0;
            Bs0[ty][tx+BLOCK_SIZE] = ((kk*BLOCK_SIZE+ty) < N  && J1<N) ? B[(kk*BLOCK_SIZE+ty)*N + J1] : 0;
            Bs0[ty][tx+2*BLOCK_SIZE] = ((kk*BLOCK_SIZE+ty) < N  && J2<N) ? B[(kk*BLOCK_SIZE+ty)*N + J2] : 0;
            Bs0[ty][tx+3*BLOCK_SIZE] = ((kk*BLOCK_SIZE+ty) < N  && J3<N) ? B[(kk*BLOCK_SIZE+ty)*N + J3] : 0;

            __syncthreads();
            #pragma unroll 
            for (unsigned int k=0;k<BLOCK_SIZE;++k) {
                Cij[0] += As0[ty][k] * Bs0[k][tx];
                Cij[1] += As0[ty][k] * Bs0[k][tx+BLOCK_SIZE];
                Cij[2] += As0[ty][k] * Bs0[k][tx+2*BLOCK_SIZE];
                Cij[3] += As0[ty][k] * Bs0[k][tx+3*BLOCK_SIZE];
                Cij[4] += As0[ty+BLOCK_SIZE][k] * Bs0[k][tx];
                Cij[5] += As0[ty+BLOCK_SIZE][k] * Bs0[k][tx+BLOCK_SIZE];
                Cij[6] += As0[ty+BLOCK_SIZE][k] * Bs0[k][tx+2*BLOCK_SIZE];
                Cij[7] += As0[ty+BLOCK_SIZE][k] * Bs0[k][tx+3*BLOCK_SIZE];
                Cij[8] += As0[ty+2*BLOCK_SIZE][k] * Bs0[k][tx];
                Cij[9] += As0[ty+2*BLOCK_SIZE][k] * Bs0[k][tx+BLOCK_SIZE];
                Cij[10] += As0[ty+2*BLOCK_SIZE][k] * Bs0[k][tx+2*BLOCK_SIZE];
                Cij[11] += As0[ty+2*BLOCK_SIZE][k] * Bs0[k][tx+3*BLOCK_SIZE];
                Cij[12] += As0[ty+3*BLOCK_SIZE][k] * Bs0[k][tx];
                Cij[13] += As0[ty+3*BLOCK_SIZE][k] * Bs0[k][tx+BLOCK_SIZE];
                Cij[14] += As0[ty+3*BLOCK_SIZE][k] * Bs0[k][tx+2*BLOCK_SIZE];
                Cij[15] += As0[ty+3*BLOCK_SIZE][k] * Bs0[k][tx+3*BLOCK_SIZE];
            }
            __syncthreads();
        }

        if(I0<N && J0<N)
          C[I0*N+J0] = Cij[0];
        if(I0<N && J1<N)
          C[I0*N+J1] = Cij[1];
        if(I0<N && J2<N)
          C[I0*N+J2] = Cij[2];
        if(I0<N && J3<N)
          C[I0*N+J3] = Cij[3];

        if(I1<N && J0<N)
          C[I1*N+J0] = Cij[4];
        if(I1<N && J1<N)
          C[I1*N+J1] = Cij[5];
        if(I1<N && J2<N)
          C[I1*N+J2] = Cij[6];
        if(I1<N && J3<N)
          C[I1*N+J3] = Cij[7];

        if(I2<N && J0<N)
          C[I2*N+J0] = Cij[8];
        if(I2<N && J1<N)
          C[I2*N+J1] = Cij[9];
        if(I2<N && J2<N)
          C[I2*N+J2] = Cij[10];
        if(I2<N && J3<N)
          C[I2*N+J3] = Cij[11];

        if(I3<N && J0<N)
          C[I3*N+J0] = Cij[12];
        if(I3<N && J1<N)
          C[I3*N+J1] = Cij[13];
        if(I3<N && J2<N)
          C[I3*N+J2] = Cij[14];
        if(I3<N && J3<N)
          C[I3*N+J3] = Cij[15];
    }
}
