// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include <algorithm>
#include "utils.h"
#include "types.h"
using namespace std;

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
    const int TK = 32;
    __shared__ _DOUBLE_ As[TK][TK], Bs[TK][TK];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int by = blockIdx.y;
    int bx = blockIdx.x;

    int I = by*TK+ty;
    int J = bx*TK+tx;
    
    _DOUBLE_ Cij = 0;

    int nTK = N/TK;
    for (int kk=0;kk<nTK;++kk) {
        As[ty][tx] = A[I*N + kk*TK+tx];
        Bs[ty][tx] = B[(kk*TK+ty)*N + J];
        __syncthreads();
        for (int k=0;k<TK;++k) {
            Cij += As[ty][k] * Bs[k][tx]; 
        }
        __syncthreads();
    }
    C[I*N+J] = Cij;
}
