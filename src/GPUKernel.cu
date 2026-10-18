#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <windows.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

constexpr int BLOCK_TILE = 16; // Same as blockDim, MUST be SQUARE
constexpr int THREAD_TILE = 4; // MUST BE A factor of BLOCK_TILE

// A, B, C - pointers in memory, where AxB = C
// A = nxm
// B = mxk
// C = nxk
// Stride A = m
// Stride B = k
// Stride C = k


//Improvement ORDER
//Naive
//Switching X and Y (global memory coalescing)
//Shared Memory Usage
//Tiling (arithmetic complexity)
//Isolation of Kernel Work (Load count vs Store count)
//Tuning

//Remaining Issues
// Bank conflicts - fix by resizing block/thread tile
// Vectorized ops - 1d Loops
// Write coalescing - Need to change computation order/add strided complexity, prevents vectorized ops.

__global__ void matmul_kernel(const float *A, const float *B, float *C, int n, int m, int k) {

    //Each block will compute blockDim^2 * threadTile^2 elements

    const int thread_row = threadIdx.y; //threadIdx.y increments second fastest (loops back after threadIdx.x hits BLOCKTILE;  y = floor(x/BLOCKTILE) )
    //Since rows are non-consecutive in row-major form, we want threadIdx.y to access rows.
    const int thread_col = threadIdx.x; //threadIdx.x increments the fastest. If we use threadIdx.x to access consecutive elements, can happen in one memory access (coalescing)
    //Since cols are consecutive in row major form, we want threadIdx.x to access cols (coaleced, in one access)

    const int global_row = blockIdx.y * (blockDim.y * THREAD_TILE) + (thread_row * THREAD_TILE);
    //The starting row index for the THREAD_TILE * THREAD_TILE set of elements this thread will compute.
    const int global_col = blockIdx.x * (blockDim.x * THREAD_TILE) + (thread_col * THREAD_TILE);
    //The starting col index for the THREAD_TILE * THREAD_TILE set of elements this thread will compute.

    //Amount of elements each thread LOADS (into shared mem): THREAD_TILE * M_FACTOR
    //Amount of elements each thread COMPUTES: THREAD_TILE * THREAD_TILE

    const int M_FACTOR = 1;
    const int S_TILE = BLOCK_TILE * THREAD_TILE; // MUST be a multiple of BLOCK_TILE
    const int M_TILE = BLOCK_TILE * M_FACTOR; // MUST be a multiple of BLOCK_TILE - To allow each thread to work

    __shared__ float Ashared[S_TILE][M_TILE]; // SMEM Caching for Block tiles
    __shared__ float Bshared[M_TILE][S_TILE];

    float accum[THREAD_TILE * THREAD_TILE]; // Register level thread tiles

    for (int i = 0; i < THREAD_TILE * THREAD_TILE; ++i) {
        accum[i] = 0;
    }

    for (int m_tile = 0; m_tile < m; m_tile += M_TILE) { //Non SIMT Loop - will occurr for ALL threads due to sync

        //Since A and B are  BLOCK_TILE^2 * THREAD_TILE * M_FACTOR, Each thead will load S_TILE rows and M_TILE cols of A (BLOCK_TILE^2 threads)
        //Possible improvment could be flattening the loops to be 1d, decoupling from BLOCK_TILE completely. This involves bounds checking with / and %

        for (int r_shift = 0; r_shift < THREAD_TILE; ++r_shift) {
            for (int c_shift = 0; c_shift < M_FACTOR; ++c_shift) {
                int As_row = thread_row + (r_shift * BLOCK_TILE); //As Rows = BLOCK_TILE * THREAD_TILE
                int As_col = thread_col + (c_shift * BLOCK_TILE); //As Cols = BLOCK_TILE * M_FACTOR
                int A_row = blockIdx.y * S_TILE + As_row;
                int A_col = m_tile + As_col;
                
                if (A_row < n && A_col < m) {
                    Ashared[As_row][As_col] = A[A_row * m + A_col];
                } else {
                    Ashared[As_row][As_col] = 0;
                }
            }
        }

        //Load Bshared
        for (int r_shift = 0; r_shift < M_FACTOR; ++r_shift) {
            for (int c_shift = 0; c_shift < THREAD_TILE; ++c_shift) {
                int Bs_row = thread_row + (r_shift * BLOCK_TILE); //Bs Rows = BLOCK_TILE * M_FACTOR
                int Bs_col = thread_col + (c_shift * BLOCK_TILE); //Bs Cols = BLOCK_TILE * THREAD_TILE
                int B_row = m_tile + Bs_row;
                int B_col = blockIdx.x * S_TILE + Bs_col;

                if (B_row < m && B_col < k) {
                    Bshared[Bs_row][Bs_col] = B[B_row * k + B_col];
                } else {
                    Bshared[Bs_row][Bs_col] = 0;
                }
            }
        }

        __syncthreads();

        for (int i = 0; i < M_TILE; ++i) {
            float a_reg[THREAD_TILE];
            float b_reg[THREAD_TILE];
            
            for (int thread_idx = 0; thread_idx < THREAD_TILE; ++thread_idx) {
                a_reg[thread_idx] = Ashared[(thread_row * THREAD_TILE) + thread_idx][i];
                b_reg[thread_idx] = Bshared[i][(thread_col * THREAD_TILE) + thread_idx];
            }

            for (int thread_tile_r = 0; thread_tile_r < THREAD_TILE; ++thread_tile_r) {
                for (int thread_tile_c = 0; thread_tile_c < THREAD_TILE; ++thread_tile_c) {
                    accum[thread_tile_r * THREAD_TILE + thread_tile_c] += a_reg[thread_tile_r] * b_reg[thread_tile_c];
                }
            }
        }

        __syncthreads();
    }

    for (int thread_tile_r = global_row; thread_tile_r < global_row + THREAD_TILE; ++thread_tile_r) {
        for (int thread_tile_c = global_col; thread_tile_c < global_col + THREAD_TILE; ++thread_tile_c) {
            if (thread_tile_r < n && thread_tile_c < k) {
                C[thread_tile_r * k + thread_tile_c] = accum[(thread_tile_r - global_row) * THREAD_TILE + (thread_tile_c - global_col)];
            }
        }
    }
}

extern "C" __declspec(dllexport) void cuda_kernel(
    const float* A,
    const float* B,
    float* C,
    int n, int m, int k)
{
    float *d_A, *d_B, *d_C;

    size_t size_A = (size_t)n * m * sizeof(float);
    size_t size_B = (size_t)m * k * sizeof(float);
    size_t size_C = (size_t)n * k * sizeof(float);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_TILE, BLOCK_TILE);
    dim3 gridDim(
        (k + blockDim.x * THREAD_TILE - 1) /
            (blockDim.x * THREAD_TILE),
        (n + blockDim.y * THREAD_TILE - 1) /
            (blockDim.y * THREAD_TILE)
    );

    matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n, m, k);

    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}