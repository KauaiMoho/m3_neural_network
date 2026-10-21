#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <windows.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

constexpr int BLOCK_TILE = 16; // Same as blockDim, MUST be SQUARE
constexpr int THREAD_TILE = 2; // MUST BE A factor of BLOCK_TILE
constexpr int BLOCK_TILE_T = 16; // Same as blockDim, MUST be SQUARE
constexpr int THREAD_TILE_T = 4; // MUST BE A factor of BLOCK_TILE

// A, B, C - pointers in memory, where AxB = C
// A = nxm
// B = mxk
// C = nxk
// Stride A = m
// Stride B = k
// Stride C = k

//Fully Naive
__global__ void matmul_kernel_naive(const float *A, const float *B, float *C, int n, int m, int k) {

    const int thread_row = threadIdx.x;
    const int thread_col = threadIdx.y;

    const int global_row = blockIdx.x * blockDim.x + thread_row;
    const int global_col = blockIdx.y * blockDim.y + thread_col;

    float accum = 0;
    
    if (global_row < n && global_col < k) {

        for (int i = 0; i < m; ++i) {
            accum += A[global_row * m + i] * B[i * k + global_col];
        }

        C[global_row * k + global_col] = accum;
    }
}

//Swap row and col indexing to make memory much faster by coalescing
__global__ void matmul_kernel_coalesced(const float *A, const float *B, float *C, int n, int m, int k) {

    const int thread_row = threadIdx.y; //threadIdx.y increments second fastest (loops back after threadIdx.x hits BLOCKTILE;  y = floor(x/BLOCKTILE) )
    //Since rows are non-consecutive in row-major form, we want threadIdx.y to access rows.
    const int thread_col = threadIdx.x; //threadIdx.x increments the fastest. If we use threadIdx.x to access consecutive elements, can happen in one memory access (coalescing)
    //Since cols are consecutive in row major form, we want threadIdx.x to access cols (coaleced, in one access)

    const int global_row = blockIdx.y * blockDim.y + thread_row;
    const int global_col = blockIdx.x * blockDim.x + thread_col;

    float accum = 0;

    if (global_row < n && global_col < k) {

        for (int i = 0; i < m; ++i) {
            accum += A[global_row * m + i] * B[i * k + global_col];
        }

        C[global_row * k + global_col] = accum;
    }
}

//Adds shared memory - Loading chunks of BLOCK_TILE*BLOCK_TILE of A and B to make memory not so slow.
__global__ void matmul_kernel_smem(const float *A, const float *B, float *C, int n, int m, int k) {

    const int thread_row = threadIdx.y; //threadIdx.y increments second fastest (loops back after threadIdx.x hits BLOCKTILE;  y = floor(x/BLOCKTILE) )
    //Since rows are non-consecutive in row-major form, we want threadIdx.y to access rows.
    const int thread_col = threadIdx.x; //threadIdx.x increments the fastest. If we use threadIdx.x to access consecutive elements, can happen in one memory access (coalescing)
    //Since cols are consecutive in row major form, we want threadIdx.x to access cols (coaleced, in one access)

    const int global_row = blockIdx.y * blockDim.y + thread_row;
    const int global_col = blockIdx.x * blockDim.x + thread_col;

    __shared__ float Ashared[BLOCK_TILE][BLOCK_TILE];// SMEM Caching
    __shared__ float Bshared[BLOCK_TILE][BLOCK_TILE];

    float accum = 0;

    for (int block_tile = 0; block_tile < m; block_tile += BLOCK_TILE) { //Non SIMT Loop - will occurr for ALL threads due to sync

        int As_row = thread_row;
        int A_row = blockIdx.y * BLOCK_TILE + As_row;
        int As_col = thread_col;
        int A_col = block_tile + As_col;
            
        if (A_row < n && A_col < m) {
            Ashared[As_row][As_col] = A[A_row * m + A_col];
        } else {
            Ashared[As_row][As_col] = 0;
        }

        int Bs_row = thread_row;
        int B_row = block_tile + Bs_row;
        int Bs_col = thread_col;
        int B_col = blockIdx.x * BLOCK_TILE + Bs_col;

        if (B_row < m && B_col < k) {
            Bshared[Bs_row][Bs_col] = B[B_row * k + B_col];
        } else {
            Bshared[Bs_row][Bs_col] = 0;
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_TILE; ++i) {

            accum += Ashared[thread_row][i] * Bshared[i][thread_col];
        }

        __syncthreads();
    }

    if (global_row < n && global_col < k) {
        C[global_row * k + global_col] = accum;
    }
}

//Adds tiles to increase computation per thread - Each thread computes THREAD_TILE * THREAD_TILE
__global__ void matmul_kernel_tiled(const float *A, const float *B, float *C, int n, int m, int k) {

    //Each block will compute blockDim^2 * threadTile^2 elements

    const int thread_row = threadIdx.y; //threadIdx.y increments second fastest (loops back after threadIdx.x hits BLOCKTILE;  y = floor(x/BLOCKTILE) )
    //Since rows are non-consecutive in row-major form, we want threadIdx.y to access rows.
    const int thread_col = threadIdx.x; //threadIdx.x increments the fastest. If we use threadIdx.x to access consecutive elements, can happen in one memory access (coalescing)
    //Since cols are consecutive in row major form, we want threadIdx.x to access cols (coaleced, in one access)

    const int global_row = blockIdx.y * (blockDim.y * THREAD_TILE) + (thread_row * THREAD_TILE);
    //The starting row index for the THREAD_TILE * THREAD_TILE set of elements this thread will compute.
    const int global_col = blockIdx.x * (blockDim.x * THREAD_TILE) + (thread_col * THREAD_TILE);
    //The starting col index for the THREAD_TILE * THREAD_TILE set of elements this thread will compute.

    __shared__ float Ashared[BLOCK_TILE * THREAD_TILE][BLOCK_TILE * THREAD_TILE]; // SMEM Caching for Block tiles
    __shared__ float Bshared[BLOCK_TILE * THREAD_TILE][BLOCK_TILE * THREAD_TILE]; //Fixed shared mem bank conflicts

    float accum[THREAD_TILE * THREAD_TILE]; // Register level thread tiles

    for (int i = 0; i < THREAD_TILE * THREAD_TILE; ++i) {
        accum[i] = 0;
    }

    for (int block_tile = 0; block_tile < m; block_tile += BLOCK_TILE * THREAD_TILE) { //Non SIMT Loop - will occurr for ALL threads due to sync

        for (int r_shift = 0; r_shift < THREAD_TILE; ++r_shift) {

            int As_row = thread_row + (r_shift * BLOCK_TILE);
            int A_row = blockIdx.y * BLOCK_TILE * THREAD_TILE + As_row;

            for (int c_shift = 0; c_shift < THREAD_TILE; ++c_shift) {

                int As_col = thread_col + (c_shift * BLOCK_TILE);
                int A_col = block_tile + As_col;
                
                if (A_row < n && A_col < m) {
                    Ashared[As_row][As_col] = A[A_row * m + A_col];
                } else {
                    Ashared[As_row][As_col] = 0;
                }
            }
        }

        for (int r_shift = 0; r_shift < THREAD_TILE; ++r_shift) {

            int Bs_row = thread_row + (r_shift * BLOCK_TILE);
            int B_row = block_tile + Bs_row;

            for (int c_shift = 0; c_shift < THREAD_TILE; ++c_shift) {
                
                int Bs_col = thread_col + (c_shift * BLOCK_TILE);
                int B_col = blockIdx.x * BLOCK_TILE * THREAD_TILE + Bs_col;

                if (B_row < m && B_col < k) {
                    Bshared[Bs_row][Bs_col] = B[B_row * k + B_col];
                } else {
                    Bshared[Bs_row][Bs_col] = 0;
                }
            }
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_TILE * THREAD_TILE; ++i) {
            for (int thread_tile_r = 0; thread_tile_r < THREAD_TILE; ++thread_tile_r) {
                for (int thread_tile_c = 0; thread_tile_c < THREAD_TILE; ++thread_tile_c) {
                    accum[thread_tile_r * THREAD_TILE + thread_tile_c] += 
                        Ashared[(thread_row * THREAD_TILE) + thread_tile_r][i] * Bshared[i][(thread_col * THREAD_TILE) + thread_tile_c];
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

//Adds registers for computation step to improve speed
__global__ void matmul_kernel_register(const float *A, const float *B, float *C, int n, int m, int k) {

    //Each block will compute blockDim^2 * threadTile^2 elements

    const int thread_row = threadIdx.y; //threadIdx.y increments second fastest (loops back after threadIdx.x hits BLOCKTILE;  y = floor(x/BLOCKTILE) )
    //Since rows are non-consecutive in row-major form, we want threadIdx.y to access rows.
    const int thread_col = threadIdx.x; //threadIdx.x increments the fastest. If we use threadIdx.x to access consecutive elements, can happen in one memory access (coalescing)
    //Since cols are consecutive in row major form, we want threadIdx.x to access cols (coaleced, in one access)

    const int global_row = blockIdx.y * (blockDim.y * THREAD_TILE) + (thread_row * THREAD_TILE);
    //The starting row index for the THREAD_TILE * THREAD_TILE set of elements this thread will compute.
    const int global_col = blockIdx.x * (blockDim.x * THREAD_TILE) + (thread_col * THREAD_TILE);
    //The starting col index for the THREAD_TILE * THREAD_TILE set of elements this thread will compute.

    __shared__ float Ashared[BLOCK_TILE * THREAD_TILE][BLOCK_TILE * THREAD_TILE]; // SMEM Caching for Block tiles
    __shared__ float Bshared[BLOCK_TILE * THREAD_TILE][BLOCK_TILE * THREAD_TILE]; //Fixed shared mem bank conflicts

    float accum[THREAD_TILE * THREAD_TILE]; // Register level thread tiles

    for (int i = 0; i < THREAD_TILE * THREAD_TILE; ++i) {
        accum[i] = 0;
    }

    for (int block_tile = 0; block_tile < m; block_tile += BLOCK_TILE * THREAD_TILE) { //Non SIMT Loop - will occurr for ALL threads due to sync

        for (int r_shift = 0; r_shift < THREAD_TILE; ++r_shift) {

            int As_row = thread_row + (r_shift * BLOCK_TILE);
            int A_row = blockIdx.y * BLOCK_TILE * THREAD_TILE + As_row;

            for (int c_shift = 0; c_shift < THREAD_TILE; ++c_shift) {

                int As_col = thread_col + (c_shift * BLOCK_TILE);
                int A_col = block_tile + As_col;
                
                if (A_row < n && A_col < m) {
                    Ashared[As_row][As_col] = A[A_row * m + A_col];
                } else {
                    Ashared[As_row][As_col] = 0;
                }
            }
        }

        for (int r_shift = 0; r_shift < THREAD_TILE; ++r_shift) {

            int Bs_row = thread_row + (r_shift * BLOCK_TILE);
            int B_row = block_tile + Bs_row;

            for (int c_shift = 0; c_shift < THREAD_TILE; ++c_shift) {
                
                int Bs_col = thread_col + (c_shift * BLOCK_TILE);
                int B_col = blockIdx.x * BLOCK_TILE * THREAD_TILE + Bs_col;

                if (B_row < m && B_col < k) {
                    Bshared[Bs_row][Bs_col] = B[B_row * k + B_col];
                } else {
                    Bshared[Bs_row][Bs_col] = 0;
                }
            }
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_TILE * THREAD_TILE; ++i) {
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

//Interesting: Coalescing Stores performs WORSE!? - from 8 to 6 TFLOPS Find out why.
__global__ void matmul_kernel_write_coalesced(const float *A, const float *B, float *C, int n, int m, int k) {

    //Each block will compute blockDim^2 * threadTile^2 elements

    const int thread_row = threadIdx.y; //threadIdx.y increments second fastest (loops back after threadIdx.x hits BLOCKTILE;  y = floor(x/BLOCKTILE) )
    //Since rows are non-consecutive in row-major form, we want threadIdx.y to access rows.
    const int thread_col = threadIdx.x; //threadIdx.x increments the fastest. If we use threadIdx.x to access consecutive elements, can happen in one memory access (coalescing)
    //Since cols are consecutive in row major form, we want threadIdx.x to access cols (coaleced, in one access)

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

            int As_row = thread_row + (r_shift * BLOCK_TILE); //As Rows = BLOCK_TILE * THREAD_TILE
            int A_row = blockIdx.y * S_TILE + As_row;

            for (int c_shift = 0; c_shift < M_FACTOR; ++c_shift) {
                
                int As_col = thread_col + (c_shift * BLOCK_TILE); //As Cols = BLOCK_TILE * M_FACTOR
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

            int Bs_row = thread_row + (r_shift * BLOCK_TILE); //Bs Rows = BLOCK_TILE * M_FACTOR
            int B_row = m_tile + Bs_row;

            for (int c_shift = 0; c_shift < THREAD_TILE; ++c_shift) {
                
                int Bs_col = thread_col + (c_shift * BLOCK_TILE); //Bs Cols = BLOCK_TILE * THREAD_TILE
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
            for (int thread_tile_r = 0; thread_tile_r < THREAD_TILE; ++thread_tile_r) {
                for (int thread_tile_c = 0; thread_tile_c < THREAD_TILE; ++thread_tile_c) {
                    accum[thread_tile_r * THREAD_TILE + thread_tile_c] +=
                        Ashared[thread_row + (thread_tile_r * BLOCK_TILE)][i] * Bshared[i][thread_col + (thread_tile_c * BLOCK_TILE)];
                }
            }
        }

        __syncthreads();
    }

    for (int r_shift = 0; r_shift < THREAD_TILE; ++r_shift) {

        int accum_row = r_shift;
        int C_row = blockIdx.y * (blockDim.y * THREAD_TILE) + thread_row + (accum_row * BLOCK_TILE);

        for (int c_shift = 0; c_shift < THREAD_TILE; ++c_shift) {
            
            int accum_col = c_shift;
            int C_col = blockIdx.x * (blockDim.x * THREAD_TILE) + thread_col + (accum_col * BLOCK_TILE);

            if (C_row < n && C_col < k) {
                C[C_row * k + C_col] = accum[accum_row * THREAD_TILE + accum_col];
            }
        }
    }
}

//Isolation of Kernel Work (Load count vs Store count) (Making Ashared and Bshared non-square), Tuning
// Have bank conflicts ? - fix by resizing block/thread tile - CHECK Nsight Compute
__global__ void matmul_kernel_final(const float *A, const float *B, float *C, int n, int m, int k) {

    //Each block will compute blockDim^2 * threadTile^2 elements

    const int thread_row = threadIdx.y; //threadIdx.y increments second fastest (loops back after threadIdx.x hits BLOCKTILE;  y = floor(x/BLOCKTILE) )
    //Since rows are non-consecutive in row-major form, we want threadIdx.y to access rows.
    const int thread_col = threadIdx.x; //threadIdx.x increments the fastest. If we use threadIdx.x to access consecutive elements, can happen in one memory access (coalescing)
    //Since cols are consecutive in row major form, we want threadIdx.x to access cols (coaleced, in one access)

    const int global_row = blockIdx.y * (blockDim.y * THREAD_TILE_T) + (thread_row * THREAD_TILE_T);
    //The starting row index for the THREAD_TILE * THREAD_TILE set of elements this thread will compute.
    const int global_col = blockIdx.x * (blockDim.x * THREAD_TILE_T) + (thread_col * THREAD_TILE_T);
    //The starting col index for the THREAD_TILE * THREAD_TILE set of elements this thread will compute.

    //Amount of elements each thread LOADS (into shared mem): THREAD_TILE * M_FACTOR
    //Amount of elements each thread COMPUTES: THREAD_TILE * THREAD_TILE

    const int M_FACTOR = 4;
    const int S_TILE = BLOCK_TILE_T * THREAD_TILE_T; // MUST be a multiple of BLOCK_TILE
    const int M_TILE = BLOCK_TILE_T * M_FACTOR; // MUST be a multiple of BLOCK_TILE - To allow each thread to work

    __shared__ float Ashared[S_TILE][M_TILE]; // SMEM Caching for Block tiles
    __shared__ float Bshared[M_TILE][S_TILE];

    float accum[THREAD_TILE_T * THREAD_TILE_T]; // Register level thread tiles

    for (int i = 0; i < THREAD_TILE_T * THREAD_TILE_T; ++i) {
        accum[i] = 0;
    }

    for (int m_tile = 0; m_tile < m; m_tile += M_TILE) { //Non SIMT Loop - will occurr for ALL threads due to sync

        //Since A and B are  BLOCK_TILE^2 * THREAD_TILE * M_FACTOR, Each thead will load S_TILE rows and M_TILE cols of A (BLOCK_TILE^2 threads)
        //Possible improvment could be flattening the loops to be 1d, decoupling from BLOCK_TILE completely. This involves bounds checking with / and %

        for (int r_shift = 0; r_shift < THREAD_TILE_T; ++r_shift) {

            int As_row = thread_row + (r_shift * blockDim.y); //As Rows = BLOCK_TILE * THREAD_TILE
            int A_row = blockIdx.y * S_TILE + As_row;

            for (int c_shift = 0; c_shift < M_FACTOR; ++c_shift) {
                
                int As_col = thread_col + (c_shift * blockDim.x); //As Cols = BLOCK_TILE * M_FACTOR
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
            
            int Bs_row = thread_row + (r_shift * blockDim.y); //Bs Rows = BLOCK_TILE * M_FACTOR
            int B_row = m_tile + Bs_row;

            for (int c_shift = 0; c_shift < THREAD_TILE_T; ++c_shift) {
               
                int Bs_col = thread_col + (c_shift * blockDim.x); //Bs Cols = BLOCK_TILE * THREAD_TILE
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
            for (int thread_tile_r = 0; thread_tile_r < THREAD_TILE_T; ++thread_tile_r) {
                for (int thread_tile_c = 0; thread_tile_c < THREAD_TILE_T; ++thread_tile_c) {
                    accum[thread_tile_r * THREAD_TILE_T + thread_tile_c] += 
                        Ashared[(thread_row * THREAD_TILE_T) + thread_tile_r][i] * Bshared[i][(thread_col * THREAD_TILE_T) + thread_tile_c];
                }
            }
        }
        __syncthreads();
    }

    for (int thread_tile_r = global_row; thread_tile_r < global_row + THREAD_TILE_T; ++thread_tile_r) {
        for (int thread_tile_c = global_col; thread_tile_c < global_col + THREAD_TILE_T; ++thread_tile_c) {
            if (thread_tile_r < n && thread_tile_c < k) {
                C[thread_tile_r * k + thread_tile_c] = accum[(thread_tile_r - global_row) * THREAD_TILE_T + (thread_tile_c - global_col)];
            }
        }
    }
}

//Remaining Issues
//Vectorized ops - 1d Loops

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
    dim3 blockDimFinal(BLOCK_TILE_T, BLOCK_TILE_T);

    dim3 gridDimNaive(
        (n + blockDim.x - 1) / blockDim.x,
        (k + blockDim.y - 1) / blockDim.y
    );
    dim3 gridDimThreadSingle(
        (k + blockDim.x - 1) / blockDim.x,
        (n + blockDim.y - 1) / blockDim.y
    );
    dim3 gridDimThreadTile(
        (k + blockDim.x * THREAD_TILE - 1) /
            (blockDim.x * THREAD_TILE),
        (n + blockDim.y * THREAD_TILE - 1) /
            (blockDim.y * THREAD_TILE)
    );
    dim3 gridDimThreadTileFinal(
        (k + blockDimFinal.x * THREAD_TILE_T - 1) /
            (blockDimFinal.x * THREAD_TILE_T),
        (n + blockDimFinal.y * THREAD_TILE_T - 1) /
            (blockDimFinal.y * THREAD_TILE_T)
    );

    //matmul_kernel_naive<<<gridDimNaive, blockDim>>>(d_A, d_B, d_C, n, m, k);
    //matmul_kernel_coalesced<<<gridDimThreadSingle, blockDim>>>(d_A, d_B, d_C, n, m, k);
    //matmul_kernel_smem<<<gridDimThreadSingle, blockDim>>>(d_A, d_B, d_C, n, m, k);
    //matmul_kernel_tiled<<<gridDimThreadTile, blockDim>>>(d_A, d_B, d_C, n, m, k);
    //matmul_kernel_register<<<gridDimThreadTile, blockDim>>>(d_A, d_B, d_C, n, m, k);
    matmul_kernel_write_coalesced<<<gridDimThreadTile, blockDim>>>(d_A, d_B, d_C, n, m, k);
    //matmul_kernel_final<<<gridDimThreadTileFinal, blockDimFinal>>>(d_A, d_B, d_C, n, m, k);

    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}