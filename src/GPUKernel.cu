#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <windows.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

constexpr int BLOCK_TILE = 16; // Same as blockDim, MUST be SQUARE
constexpr int THREAD_TILE = 2; // MUST BE A factor of BLOCK_TILE

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
        for (int thread_tile_r = 0; thread_tile_r < THREAD_TILE; ++thread_tile_r) {
            for (int thread_tile_c = 0; thread_tile_c < THREAD_TILE; ++thread_tile_c) {
                for (int i = 0; i < BLOCK_TILE * THREAD_TILE; ++i) {
                    accum[thread_tile_r * THREAD_TILE + thread_tile_c] += Ashared[r_block_idx + thread_tile_r][i] * Bshared[i][c_block_idx + thread_tile_c];
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

void matmul_cpu(const float *A, const float *B, float *C, int n, int m, int k) {
    for (int i = 0; i < n; ++i) {
        for (int l = 0; l < k; ++l) {
            float sum = 0;
            for (int j = 0; j < m; ++j) {
                sum += A[i * m + j] * B[j * k + l];
            }
            C[i * k + l] = sum;
        }
    }
}

void fill_random(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = (float)(rand() % 100) / 10.0;
    }
}

bool verify_results(const float *gpu_result, const float *cpu_result, int size, float tolerance = 0.01) {

    int error = false;

    for (int i = 0; i < size; ++i) {
        float diff = fabsf(gpu_result[i] - cpu_result[i]);
        if (diff > tolerance) {
            printf("Mismatch at index %d: GPU=%.6f, CPU=%.6f, diff=%.6f\n", i, gpu_result[i], cpu_result[i], diff);
            error = true;
        }
    }

    return error;
}

double get_time_ms() {

    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / frequency.QuadPart;
}

int run_test() {

    printf("CUDA Matmul Test\n\n");

    srand((unsigned int)time(NULL));

    int test_cases[][3] = {
        {64, 64, 64},
        {128, 128, 128},
        {512, 128, 512},
        {1024, 512, 1024},
    };

    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);

    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    for (int test = 0; test < num_tests; ++test) {
        int n = test_cases[test][0];
        int m = test_cases[test][1];
        int k = test_cases[test][2];

        printf("Test %d: Matrix sizes A(%dx%d) x B(%dx%d) = C(%dx%d)\n", test + 1, n, m, m, k, n, k);

        size_t size_A = n * m * sizeof(float);
        size_t size_B = m * k * sizeof(float);
        size_t size_C = n * k * sizeof(float);

        float *h_A = (float *)malloc(size_A);
        float *h_B = (float *)malloc(size_B);
        float *h_C_gpu = (float *)malloc(size_C);
        float *h_C_cpu = (float *)malloc(size_C);

        if (!h_A || !h_B || !h_C_gpu || !h_C_cpu) {
            fprintf(stderr, "Failed to allocate memory\n");
            return 1;
        }

        fill_random(h_A, n, m);
        fill_random(h_B, m, k);

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size_A);
        cudaMalloc(&d_B, size_B);
        cudaMalloc(&d_C, size_C);

        cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

        dim3 blockDim(BLOCK_TILE, BLOCK_TILE);
        dim3 gridDim((k + blockDim.x * THREAD_TILE - 1) / (blockDim.x * THREAD_TILE), (n + blockDim.y * THREAD_TILE - 1) / (blockDim.y * THREAD_TILE));

        printf("Grid: (%d, %d), Block: (%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

        matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n, m, k);
        cudaDeviceSynchronize();

        double start_gpu = get_time_ms();
        int num_iterations = 5;

        for (int i = 0; i < num_iterations; ++i) {
            matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n, m, k);
        }
        cudaDeviceSynchronize();
        double end_gpu = get_time_ms();
        double gpu_time = (end_gpu - start_gpu) / num_iterations;

        cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

        double start_cpu = get_time_ms();
        matmul_cpu(h_A, h_B, h_C_cpu, n, m, k);
        double end_cpu = get_time_ms();
        double cpu_time = end_cpu - start_cpu;

        bool error = verify_results(h_C_gpu, h_C_cpu, n*k);

        double gflops = (2*n*m*k) / (gpu_time*1000000);
        double speedup = cpu_time / gpu_time;

        printf("GPU Time: %.3f ms\n", gpu_time);
        printf("CPU Time: %.3f ms\n", cpu_time);
        printf("Speedup: %.2fx\n", speedup);
        printf("Performance: %.2f GFLOPS\n", gflops);
        printf("Result: %s\n\n", error ? "FAILED" : "PASSED");

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C_gpu);
        free(h_C_cpu);
    }

    printf("FINISHED!\n");
    return 0;
}

int main() {
    return run_test();
}

// Extern from Matrix Class
// TODO: Handle host/device memory transfer, setup tuning vars in future, call matmul_kernel.
extern "C" void matmul_cuda(
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