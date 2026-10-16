#ifndef MATRIX_H
#define MATRIX_H
#include <neon_shim.h>
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <tuple>
#include "AMXKernel.h"

class Tensor {
 private:
  // flattened approach
  uint32_t dim_len;
  uint32_t data_len;
  uint32_t aligned_data_len;
  int* dims;
  int* dists;
  float* data;
  static uint16_t tile;  // MUST be a multiple of 4.
  static constexpr uint16_t alignment =
      64;  // 16 minimum for SIMD optimization, can change to that, aligning to
           // a higher multiple of 16 can reduce cache line splits, however in
           // my testing this dosent have much of an effect on speed.
           // 64 bit alignment for AMX padding, can be 32 for regular SIMD ops

  

  Tensor(int* dims_n, int dim_len, float* data_n,
         bool copy);  // Create a new matrix with given data, can choose to copy
                      // or take ownership
  Tensor(int* dims_n, int dim_len, float* data_n, int data_len,
         int* dists);  // Strictly for direct cloning, use incase view has
                       // changed (reshape/broadcast).
  int convert_idx(const std::initializer_list<int>& pos)
      const;  // Convert given index to 1d flattend index using strides

  std::tuple<int, int, int> get_matmul_tile(int n, int m, int k)
      const;  // Get a tile size for a specific matix size. It will assume
              // matrix dimensions are square for fitting into cache for
              // simplicity.
  #ifdef WINDOWS_VER
  void matmul_cuda(const float* A, const float* B, float* C, int n, int m,
                   int k) const;
  #endif
  void matmul_cpu_batched(const float* __restrict__ A,
                          const float* __restrict__ B, float* __restrict__ C,
                          const int* this_dists, const int* other_dists, int n,
                          int m, int k, int z) const;
  void matmul_cpu_outer(const float* __restrict__ A,
                        const float* __restrict__ B, float* __restrict__ C,
                        int n, int m, int k) const;
  void matmul_cpu_unrolled_16x(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C, int n, int m,
                               int k) const;
  void matmul_cpu_unrolled_8x(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C, int n, int m, int k) const;
  void matmul_cpu_unrolled_4x(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C, int n, int m, int k) const;
  #ifndef WINDOWS_VER
  void matmul_amx(const float* __restrict__ A, const float* __restrict__ B,
                  float* __restrict__ C, int n, int m, int k) const;
  #endif
  void matmul_cpu(const float* __restrict__ A, const float* __restrict__ B,
                  float* __restrict__ C, int n, int m, int k) const;
  void matmul_cpu_tiled(const float* __restrict__ A,
                        const float* __restrict__ B, float* __restrict__ C,
                        int n, int m, int k) const;
  void matmul_cpu_tiled_old(const float* __restrict__ A,
                            const float* __restrict__ B, float* __restrict__ C,
                            int n, int m, int k) const;
  void matmul_cpu_naive(const float* __restrict__ A,
                        const float* __restrict__ B, float* __restrict__ C,
                        int n, int m, int k) const;  // For comparison.

  // Will transpose a matrix physically using simd operations, used internally
  // for efficient matmul and public facing transpose2d.
  void simd_transpose(const float* A, float* B, int n, int m, int z = 0,
                      const int* dists_new = nullptr) const;
  int* get_dims_clone() const;   // Get a deep copy of dims.
  int* get_dists_clone() const;  // Get a deep copy of dists array.
  int* get_broadcasted_strides(const int* dims_new, int dim_len_new)
      const;  // Get the broadcasted strides for this given a set of dimensions

  float* default_data_alloc()
      const;  // Allocate an aligned float array based on aligned_data_len.
  float* init_data_alloc(size_t size);  // Allocate an aligned float array and
                                        // initialize aligned_data_len.
  float* float_size_alloc(
      size_t size) const;  // Allocate an aligned float array.
  int* int_size_alloc(
      size_t size) const;  // Allocate an unaligned integer array.

  void print_array(const float* arr, int len,
                   int max) const;  // Print a given float array
  void print_array(const int* arr, int len,
                   int max) const;  // Print a given int array

 public:
  Tensor(const int* dims_n, int dim_len,
         const float* data_n);  // Create a new matrix with a given data array
                                // (flattened, row major), will copy data
  Tensor(const int* dims_n, int dim_len,
         float val);  // Create a new matrix filled with a given float
  Tensor(
      const int* dims_n, int dim_len,
      unsigned int random_seed = 0);  // Create a new matrix filled with random
                                      // floats between [0-1), can set seed
  Tensor(const Tensor& other);                 // Copy constructor
  Tensor& operator=(const Tensor& other);      // Copy assignment operator
  Tensor(Tensor&& other) noexcept;             // Move constructor
  Tensor& operator=(Tensor&& other) noexcept;  // Move assignment operator
  ~Tensor();                                   // Destructor
  Tensor matmul(const Tensor& other)
      const;             // Matmul, extensive docs in source and usage guides.
  Tensor clone() const;  // Return a deep copy clone of this object
  Tensor scmul(float s)
      const;  // Will multiply matrix by a scalar, and return new matrix.
  Tensor emul(const Tensor& other)
      const;  // Will multiply two matrices elementwise, and return new matrix.
              // Will prioritize row-col multiplication over col-row
              // multiplication for 1D case
  Tensor add(const Tensor& other)
      const;  // Will add two matrices, and return new matrix. Will prioritize
              // row-col multiplication over col-row multiplication for 1D case
  Tensor apply(float (*func)(
      float)) const;  // Will apply a given function, and return new matrix.
  Tensor transpose2d() const;   // Will transpose data physically leveraging
                                // simd, and return new tranposed matrix.
  void scmul_inplace(float s);  // Will multiply matrix by a scalar inplace.
  void emul_inplace(
      const Tensor& other);  // Will multiply two matrices elementwise inplace.
  void add_inplace(const Tensor& other);  // Will add two matrices inplace.
  void apply_inplace(
      float (*func)(float));  // Will apply a given function inplace.
  Tensor sum_rows() const;    // Sum all rows into one row vector.
  Tensor sum_cols() const;    // Sum all cols into one col vector.
  float sum() const;          // Sum all rows and cols into one float.
  float get(const std::initializer_list<int>& pos)
      const;  // Get a value using format {x, y, z, ...}
  float get_index(int i)
      const;  // Get a value using a flattened index (mostly internal use)
  void set(const std::initializer_list<int>& pos,
           float val);  // Set a value using format {x, y, z, ...}
  void set_index(
      int i,
      float val);  // Set a value using a flattened index (mostly internal use)
  int get_dims_index(int i) const;  // Get dims using an flattened index
  int get_dim_len() const;  // Get length of all dimensions (returns 3 for 3d)
  void print_dims(int max = 50) const;   // Print dims
  void print_dists(int max = 50) const;  // Print dists
  void print_data(
      int max = 50) const;  // Print data values up to a given max, default 50
  static void set_tile(int t);
  static int get_tile();
  static int get_alignment();
};

extern "C" void cuda_kernel(const float* A, const float* B, float* C, int n,
                            int m, int k);

#endif