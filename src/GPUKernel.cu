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

// TODO: Roofline analysis, FLOP calculator
__global__ void matmul_kernel(const float *A, const float *B, float *C, int n, int m, int k) {

    const int r_block_idx = threadIdx.y * THREAD_TILE;
    const int c_block_idx = threadIdx.x * THREAD_TILE;

    const int r_global_idx = blockIdx.y * (blockDim.y * THREAD_TILE) + r_block_idx; // Swap to make use of NVIDIA global coalescing (SIMT).
    const int c_global_idx = blockIdx.x * (blockDim.x * THREAD_TILE) + c_block_idx;

    __shared__ float Ashared[BLOCK_TILE * THREAD_TILE][BLOCK_TILE * THREAD_TILE]; // SMEM Caching for Block tiles
    __shared__ float Bshared[BLOCK_TILE * THREAD_TILE][BLOCK_TILE * THREAD_TILE];

    float accum[THREAD_TILE * THREAD_TILE]; // Register level thread tiles

    #pragma unroll
    for (int i = 0; i < THREAD_TILE * THREAD_TILE; ++i) {
        accum[i] = 0;
    }

    for (int block_tile = 0; block_tile < m; block_tile += BLOCK_TILE * THREAD_TILE) {

        int thread_tile_global_r = r_global_idx;
        int thread_tile_block_r = r_block_idx;
        int row_block_tile = block_tile + r_block_idx;
        int thread_tile_global_c = c_global_idx;
        int thread_tile_block_c = c_block_idx;
        int col_block_tile = block_tile + c_block_idx;

        for (int thread_tile_r = 0; thread_tile_r < THREAD_TILE; ++thread_tile_r) {
            for (int thread_tile_c = 0; thread_tile_c < THREAD_TILE; ++thread_tile_c) {

                if (thread_tile_global_r < n && col_block_tile < m) {
                    Ashared[thread_tile_block_r][thread_tile_block_c] = A[thread_tile_global_r * m + col_block_tile];
                } else {
                    Ashared[thread_tile_block_r][thread_tile_block_c] = 0; // 0 Padding at edges
                }

                if (thread_tile_global_c < k && row_block_tile < m){
                    Bshared[thread_tile_block_r][thread_tile_block_c] = B[row_block_tile * k + thread_tile_global_c];
                } else {
                    Bshared[thread_tile_block_r][thread_tile_block_c] = 0;
                }

                thread_tile_global_c++;
                thread_tile_block_c++;
                col_block_tile++;
            }
            thread_tile_global_r++;
            thread_tile_block_r++;
            row_block_tile++;
            thread_tile_global_c = c_global_idx;
            thread_tile_block_c = c_block_idx;
            col_block_tile = block_tile + c_block_idx;
        }

        __syncthreads();

        // Increase arithmetic intensity by computing multiple values of C per thread
        for (int i = 0; i < BLOCK_TILE * THREAD_TILE; ++i) {
            float a_reg[THREAD_TILE];
            float b_reg[THREAD_TILE];
            
            for (int thread_tile_r = 0; thread_tile_r < THREAD_TILE; ++thread_tile_r) {
                a_reg[thread_tile_r] = Ashared[r_block_idx + thread_tile_r][i];
            }
                
            for (int thread_tile_c = 0; thread_tile_c < THREAD_TILE; ++thread_tile_c) {
                b_reg[thread_tile_c] = Bshared[i][c_block_idx + thread_tile_c];
            }

            for (int thread_tile_r = 0; thread_tile_r < THREAD_TILE; ++thread_tile_r) {
                for (int thread_tile_c = 0; thread_tile_c < THREAD_TILE; ++thread_tile_c) {
                    accum[thread_tile_r * THREAD_TILE + thread_tile_c] += a_reg[thread_tile_r] * b_reg[thread_tile_c];
                }
            }
        }

        __syncthreads();
    }

    int thread_tile_global_r = r_global_idx;
    int thread_tile_global_c = c_global_idx;

    for (int thread_tile_r = 0; thread_tile_r < THREAD_TILE; ++thread_tile_r) {
        for (int thread_tile_c = 0; thread_tile_c < THREAD_TILE; ++thread_tile_c) {
            if (thread_tile_global_r < n && thread_tile_global_c < k) {
                C[thread_tile_global_r * k + thread_tile_global_c] = accum[thread_tile_r * THREAD_TILE + thread_tile_c];
            }
            thread_tile_global_c++;
        }
        thread_tile_global_r++;
        thread_tile_global_c = c_global_idx;
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