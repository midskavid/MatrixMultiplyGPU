// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include <algorithm>
#include "utils.h"
#include "types.h"
using namespace std;

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
    int i = threadIdx.y;
    int j = threadIdx.x;

    int ii = blockIdx.y;
    int jj = blockIdx.x;

    int TK = blockDim.x;
    
    _DOUBLE_ Cij = 0;
    for (int kk=0;kk<N;kk+=TK) {
        for (int k = kk;k<min(kk+TK,N);++k) {
            Cij += A[(ii*TK+i)*N + k] * B[k*N+jj*TK+j]; 
        }
    }
    C[(ii*TK+i)*N + jj*TK+j] = Cij;
}
