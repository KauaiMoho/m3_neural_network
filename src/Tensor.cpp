#include "../include/Tensor.h"

uint16_t Tensor::tile = 512;

Tensor::Tensor(const int *dims_n, int dim_len, const float *data_n)
    : dim_len(dim_len)
{
  if (dim_len == 0)
  {
    throw std::invalid_argument("Tensor dimensions cannot be empty!");
  }

  dists = int_size_alloc(dim_len);

  dims = int_size_alloc(dim_len);

  int pos = 1;
  for (int i = dim_len - 1; i > 0; --i)
  {
    dists[i] = pos;
    dims[i] = dims_n[i];
    pos *= dims[i];
  }
  dims[0] = dims_n[0];
  dists[0] = pos;
  data_len = pos * dims[0];

  // Alignment allows for data to start at a memory address aligning with cache
  // line size Allows for SIMD instructions to be faster and more efficient, as
  // all loads/stores will be within cache line boundaries, avoiding cache
  // misses The alignment size in general should be a multiple of the alignment
  // value. (this may depend on C++ version)
  data = init_data_alloc(data_len);

  for (size_t i = 0; i < data_len; ++i)
  {
    data[i] = data_n[i];
  }
}

Tensor::Tensor(int *dims_n, int dim_len, float *data_n, int data_len,
               int *dists_n)
    : dim_len(dim_len), data_len(data_len)
{
  if (dim_len == 0)
  {
    throw std::invalid_argument("Tensor dimensions cannot be empty!");
  }

  dists = int_size_alloc(dim_len);

  dims = int_size_alloc(dim_len);

  for (size_t i = 0; i < dim_len; ++i)
  {
    dists[i] = dists_n[i];
    dims[i] = dims_n[i];
  }

  data = init_data_alloc(data_len);

  for (size_t i = 0; i < data_len; ++i)
  {
    data[i] = data_n[i];
  }
}

Tensor::Tensor(int *dims_n, int dim_len, float *data_n, bool copy)
    : dim_len(dim_len)
{
  if (dim_len == 0)
  {
    throw std::invalid_argument("Tensor dimensions cannot be empty!");
  }

  if (copy)
  {
    // Same as public constructor - can change later to get of repeated code
    dists = int_size_alloc(dim_len);

    dims = int_size_alloc(dim_len);

    int pos = 1;
    for (int i = dim_len - 1; i > 0; --i)
    {
      dists[i] = pos;
      dims[i] = dims_n[i];
      pos *= dims[i];
    }
    dims[0] = dims_n[0];
    dists[0] = pos;
    data_len = pos * dims[0];

    data = init_data_alloc(data_len);

    for (size_t i = 0; i < data_len; ++i)
    {
      data[i] = data_n[i];
    }
  }
  else
  {
    dims = dims_n;

    // NEEDS TO BE ALIGNED
    data = data_n;

    dists = int_size_alloc(dim_len);

    int pos = 1;
    for (int i = dim_len - 1; i > 0; --i)
    {
      dists[i] = pos;
      pos *= dims[i];
    }
    dists[0] = pos;
    data_len = pos * dims[0];

    aligned_data_len = data_len * sizeof(float);
    size_t remainder = aligned_data_len % alignment;
    if (remainder != 0)
    {
      aligned_data_len += alignment - remainder;
    }
  }
}

Tensor::Tensor(const int *dims_n, int dim_len, float val) : dim_len(dim_len)
{
  if (dim_len == 0)
  {
    throw std::invalid_argument("Tensor dimensions cannot be empty!");
  }

  dists = int_size_alloc(dim_len);

  dims = int_size_alloc(dim_len);

  int pos = 1;
  for (int i = dim_len - 1; i > 0; --i)
  {
    dists[i] = pos;
    dims[i] = dims_n[i];
    pos *= dims[i];
  }
  dims[0] = dims_n[0];
  dists[0] = pos;
  data_len = pos * dims[0];

  data = init_data_alloc(data_len);

  for (size_t i = 0; i < data_len; ++i)
  {
    data[i] = val;
  }
}

Tensor::Tensor(const int *dims_n, int dim_len, unsigned int random_seed)
    : dim_len(dim_len)
{
  if (dim_len == 0)
  {
    throw std::invalid_argument("Tensor dimensions cannot be empty!");
  }

  if (random_seed == 0)
  {
    std::random_device rd;
    random_seed = rd();
  }
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::mt19937 gen(random_seed);

  dists = int_size_alloc(dim_len);

  dims = int_size_alloc(dim_len);

  int pos = 1;
  for (int i = dim_len - 1; i > 0; --i)
  {
    dists[i] = pos;
    dims[i] = dims_n[i];
    pos *= dims[i];
  }
  dims[0] = dims_n[0];
  dists[0] = pos;
  data_len = pos * dims[0];

  data = init_data_alloc(data_len);

  for (size_t i = 0; i < data_len; ++i)
  {
    data[i] = dist(gen);
  }
}

Tensor::Tensor(const Tensor &other)
    : dim_len(other.get_dim_len()),
      data_len(other.data_len),
      aligned_data_len(other.aligned_data_len)
{
  dists = int_size_alloc(dim_len);

  dims = int_size_alloc(dim_len);

  for (size_t i = 0; i < dim_len; ++i)
  {
    dists[i] = other.dists[i];
    dims[i] = other.dims[i];
  }

  data = default_data_alloc();

  for (size_t i = 0; i < data_len; ++i)
  {
    data[i] = other.data[i];
  }
}

Tensor &Tensor::operator=(const Tensor &other)
{
  if (this == &other)
  {
    return *this;
  }

  free(dims);
  free(dists);
  tensor_free(data);

  dim_len = other.get_dim_len();
  data_len = other.data_len;
  aligned_data_len = other.aligned_data_len;

  dists = int_size_alloc(dim_len);

  dims = int_size_alloc(dim_len);

  for (size_t i = 0; i < dim_len; ++i)
  {
    dists[i] = other.dists[i];
    dims[i] = other.dims[i];
  }

  data = default_data_alloc();

  for (size_t i = 0; i < data_len; ++i)
  {
    data[i] = other.data[i];
  }

  return *this;
}

Tensor::Tensor(Tensor &&other) noexcept
    : dim_len(other.get_dim_len()),
      data_len(other.data_len),
      aligned_data_len(other.aligned_data_len),
      dims(other.dims),
      dists(other.dists),
      data(other.data)
{
  other.dim_len = 0;
  other.data_len = 0;
  other.aligned_data_len = 0;
  other.dims = nullptr;
  other.dists = nullptr;
  other.data = nullptr;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept
{
  if (this == &other)
  {
    return *this;
  }

  free(dims);
  free(dists);
  tensor_free(data);

  dim_len = other.get_dim_len();
  data_len = other.data_len;
  aligned_data_len = other.aligned_data_len;
  dims = other.dims;
  dists = other.dists;
  data = other.data;

  other.dim_len = 0;
  other.data_len = 0;
  other.aligned_data_len = 0;
  other.dims = nullptr;
  other.dists = nullptr;
  other.data = nullptr;

  return *this;
}

Tensor::~Tensor()
{
  free(dims);
  free(dists);
  tensor_free(data);
}

float *Tensor::default_data_alloc() const
{
  float *arr = (float *)tensor_alloc(alignment, aligned_data_len);

  if (arr == nullptr)
  {
    throw std::runtime_error("Default data memory allocation error");
  }

  return arr;
}

float *Tensor::init_data_alloc(size_t size)
{
  aligned_data_len = size * sizeof(float);
  size_t remainder = aligned_data_len % alignment;
  if (remainder != 0)
  {
    aligned_data_len += alignment - remainder;
  }

  float *arr = (float *)tensor_alloc(alignment, aligned_data_len);

  if (arr == nullptr)
  {
    throw std::runtime_error("Init data memory allocation error");
  }

  return arr;
}

float *Tensor::float_size_alloc(size_t size) const
{
  size_t a_size = size * sizeof(float);
  size_t remainder = a_size % alignment;
  if (remainder != 0)
  {
    a_size += alignment - remainder;
  }

  float *arr = (float *)tensor_alloc(alignment, a_size);

  if (arr == nullptr)
  {
    throw std::runtime_error("Float memory allocation error");
  }

  return arr;
}

int *Tensor::int_size_alloc(size_t size) const
{
  int *arr = (int *)malloc(size * sizeof(int));

  if (arr == nullptr)
  {
    throw std::runtime_error("Int memory allocation error");
  }

  return arr;
}

void Tensor::print_array(const float *arr, int len, int max) const
{
  int end = std::min(len, max);
  for (size_t i = 0; i < end; ++i)
  {
    std::cout << arr[i];
    if (i < end - 1)
    {
      std::cout << ", ";
    }
  }
  std::cout << "\n";
}

void Tensor::print_array(const int *arr, int len, int max) const
{
  int end = std::min(len, max);
  for (size_t i = 0; i < end; ++i)
  {
    std::cout << arr[i];
    if (i < end - 1)
    {
      std::cout << ", ";
    }
  }
  std::cout << "\n";
}

int Tensor::convert_idx(const std::initializer_list<int> &pos) const
{
  // Converts between regular indexing (nd) and stride position (1d)
  int idx = 0;
  int i = 0;
  if (pos.size() != static_cast<size_t>(dim_len))
  {
    throw std::invalid_argument("Wrong number of indices!");
  }
  for (int value : pos)
  {
    idx += value * dists[i];
    ++i;
  }
  if (idx > data_len - 1 || idx < 0)
    throw std::invalid_argument("Invalid position dimensions!");
  return idx;
}

int *Tensor::get_broadcasted_strides(const int *dims_new,
                                     int dim_len_new) const
{
  // Gets the new strides (row major, flattened) for a given broadcast
  if (dim_len_new >= dim_len)
  {
    int *dists_new = int_size_alloc(dim_len_new);
    int diff = dim_len_new - dim_len;

    for (int i = dim_len_new - 1; i >= 0; --i)
    {
      // Get correct indexing between two sets of dimensions (dimension size may
      // be unequal)
      int i_old = i - diff;
      if (i_old < 0)
      {
        // If the index is out of bounds for the shorter dimension, set its
        // stride to 0 (copy axis along dimension)
        dists_new[i] = 0;
      }
      else
      {
        // If the dimensions match up, the strides will stay the same
        if (dims[i_old] == dims_new[i])
        {
          dists_new[i] = dists[i_old];
          // If the dimension being broadcasted is size 1, the elements can just
          // be copied along axis (stride 0)
        }
        else if (dims[i_old] == 1)
        {
          dists_new[i] = 0;
        }
        else
        {
          // Dimensions incompatible, cannot be broadcasted
          // Typically will not run since this is already checked during
          // broadcasting dimension computation, but adds fallback error
          // handling.
          free(dists_new);
          throw std::invalid_argument(
              "Incompatible dimensions for broadcasting!");
        }
      }
    }
    return dists_new;
  }
  else
  {
    throw std::invalid_argument("Invalid dimension size for broadcasting!");
  }
}

std::tuple<int, int, int> Tensor::get_matmul_tile(int n, int m, int k) const
{
  // Use loop order to optimize L Cache loading.
  // Use sysctl -a | grep cache to check Apple Silicon Cache Size
  // use sysctl -a | grep perflevel to check compute clusters

  constexpr size_t L1_bytes = 128 * 1024; // Performance cores
  constexpr size_t L2_bytes = 16 * 1024 * 1024;
  constexpr int cache_line_size = 128;

  // Choose between L1 and L2 cache based on matrix size
  size_t cache_line_floats = cache_line_size / sizeof(float);
  size_t usable_cache_bytes;
  if (n * m * k * sizeof(float) <= L1_bytes)
  { // Heuristic, very conservative, estimating based on general dimensions
    // Only use max 2/3 of the available bytes in a given cache.
    usable_cache_bytes = L1_bytes / 1.5;
  }
  else
  {
    usable_cache_bytes = L2_bytes / 1.5;
  }

  size_t usable_cache_floats = usable_cache_bytes / sizeof(float);

  // Assume T_m = m since we dont tile m for inner product.

  // How many rows of A and columns of B can we load at once so that the CPU doesn’t go to main memory too often?

  //  m * (Tn + Tk) = usable_cache_floats
  //  (Tn + Tk) = usable_cache_floats / m
  //  (( T_k * ratio) + Tk) = usable_cache_floats / m (Heuristic to solve
  //  impossible equasion)

  float ratio = static_cast<float>(n) /
                (k * 2); // Ratio will give more tile space to T_k, since it is
                         // the more important dimension for inner product.

  int T_k = static_cast<int>((usable_cache_floats / (m * (1.0 + ratio))));
  int T_n = static_cast<int>(T_k * ratio);

  // Only used in outer product, placeholder
  int T_m;
  if (m * sizeof(float) < (L2_bytes / 4))
  {
    T_m = m;
  }
  else
  {
    T_m = 1024;
  }

  // Now we round down to cache line size. Could use bit masking since cache
  // line power of 2 (originally i did this, but changed back for clarity):
  // mat_tile & ~(cache_line_floats - 1) We do this to make sure all cache line
  // data loads are perfectly used, and round down to make sure we do not exceed
  // usable cache size.
  T_n -= T_n % cache_line_floats;
  T_m -= T_m % cache_line_floats;
  T_k -= T_k % cache_line_floats;

  if (T_n <= 0)
  {
    // If tile size is small, stick with cache line float size.
    T_n = cache_line_floats;
  }

  if (T_m <= 0)
  {
    T_m = cache_line_floats;
  }

  if (T_k <= 0)
  {
    T_k = cache_line_floats;
  }

  // Now we will only work with sub matrices of size mat_tile, which is good as
  // it will allow the matmul code to always pull/write straight from CPU cache,
  // avoiding slower memory access.
  return std::make_tuple(T_n, T_m, T_k);
}

void Tensor::matmul_cpu_batched(const float *__restrict__ A,
                                const float *__restrict__ B,
                                float *__restrict__ C, const int *this_dists,
                                const int *other_dists, int n, int m, int k,
                                int z) const
{
  std::tuple<int, int, int> tile_data = get_matmul_tile(n, m, k);
  int T_n = std::get<0>(tile_data);
  int T_k = std::get<2>(tile_data);

  float *B_t = float_size_alloc(m * k);

  // We choose to transpose the data physically because reading col - major is
  // inefficient for cache, even if we just transpose the dist strides the cache
  // reading will be slow for large matrices + wanted to challenge myself to
  // write a simd transpose.

  // 2 main benefits:
  //  1. Transposing allows for contiguous reading of columns for matrix B (due
  //  to storage in row-major), which avoids cache misses.
  //  2. Contiguous reading allows for direct vectorization, as SIMD loads
  //  require it.
  simd_transpose(B, B_t, m, k, z, other_dists);
  for (size_t ic = 0; ic < n; ic += T_n)
  {
    size_t iE = std::min(ic + T_n, static_cast<size_t>(n));
    for (size_t lc = 0; lc < k; lc += T_k)
    {
      size_t lE = std::min(lc + T_k, static_cast<size_t>(k));
      for (size_t i = ic; i < iE; ++i)
      {
        const float *ptrA = &A[z * this_dists[0] + i * this_dists[1]];
        for (size_t l = lc; l < lE; ++l)
        {
          float sum = 0;
          float32x4_t acc = vdupq_n_f32(0.0f);

          const float *ptrB = &B_t[l * m];

          size_t j = 0;
          for (; j + 3 < m; j += 4)
          {
            float32x4_t a = vld1q_f32(ptrA + j * this_dists[2]);
            float32x4_t b = vld1q_f32(ptrB + j);
            acc = vfmaq_f32(acc, a, b);
          }
          for (; j < m; ++j)
          {
            sum += ptrA[j * this_dists[2]] * ptrB[j];
          }

          sum += vaddvq_f32(acc);
          C[n * k * z + i * k + l] = sum;
        }
      }
    }
  }
  tensor_free(B_t);
}

#ifdef WINDOWS_VER
void Tensor::matmul_cuda(const float *A, const float *B, float *C, int n, int m,
                         int k) const
{
  ::cuda_kernel(A, B, C, n, m, k);
}
#endif

void Tensor::matmul_cpu_outer(const float *__restrict__ A,
                              const float *__restrict__ B,
                              float *__restrict__ C, int n, int m,
                              int k) const
{
  // A = nxm
  // B = mxk
  // C = nxk
  // Stride A = m
  // Stride B = k
  // Stride C = k

  std::tuple<int, int, int> tile_data = get_matmul_tile(n, m, k);
  int T_n = std::get<0>(tile_data);
  int T_m = std::get<1>(tile_data); // shared dim
  int T_k = std::get<2>(tile_data);

  float *A_t = float_size_alloc(n * m);

  memset(C, 0, n * k * sizeof(float));

  simd_transpose(A, A_t, n, m);
  for (size_t jc = 0; jc < m; jc += T_m)
  {
    size_t jE = std::min(jc + T_m, static_cast<size_t>(m));
    for (size_t ic = 0; ic < n; ic += T_n)
    {
      size_t iE = std::min(ic + T_n, static_cast<size_t>(n));
      for (size_t lc = 0; lc < k; lc += T_k)
      {
        size_t lE = std::min(lc + T_k, static_cast<size_t>(k));

        for (size_t j = jc; j < jE; ++j)
        { // m
          for (size_t i = ic; i < iE; ++i)
          { // n

            float aVal =
                A_t[j * n +
                    i]; // Contiguous in memory now, so better cache usage.
            float32x4_t a =
                vdupq_n_f32(aVal); // get value of A, dupe across lanes
            const float *ptrB = &B[j * k + lc];
            float *ptrC = &C[i * k + lc];

            size_t l = lc;
            for (; l + 3 < lE; l += 4)
            {
              float32x4_t b = vld1q_f32(ptrB);
              float32x4_t c = vld1q_f32(ptrC);
              c = vmlaq_f32(c, a, b);
              vst1q_f32(ptrC, c);
              ptrB += 4;
              ptrC += 4;
            }

            // Scalar edge case handling
            for (; l < (lE - lc); ++l)
            {
              (*ptrC) += aVal * (*ptrB);
              ptrB += 1;
              ptrC += 1;
            }
          }
        }
      }
    }
  }
  tensor_free(A_t);
}

void Tensor::matmul_cpu_unrolled_16x(const float *__restrict__ A,
                                     const float *__restrict__ B,
                                     float *__restrict__ C, int n, int m,
                                     int k) const
{
  // A = nxm
  // B = mxk
  // C = nxk
  // Stride A = m
  // Stride B = k
  // Stride C = k

  std::tuple<int, int, int> tile_data = get_matmul_tile(n, m, k);
  int T_n = std::get<0>(tile_data);
  int T_k = std::get<2>(tile_data);

  float *B_t = float_size_alloc(m * k);

  simd_transpose(B, B_t, m, k);

  for (size_t ic = 0; ic < n; ic += T_n)
  {
    size_t iE = std::min(ic + T_n, static_cast<size_t>(n));
    for (size_t lc = 0; lc < k; lc += T_k)
    {
      size_t lE = std::min(lc + T_k, static_cast<size_t>(k));
      for (size_t i = ic; i < iE; ++i)
      {
        size_t l = lc;
        const float *ptrA = &A[i * m];

        for (; l + 15 < lE; l += 16)
        {
          float32x4_t acc0 = vdupq_n_f32(0.0f);
          float32x4_t acc1 = vdupq_n_f32(0.0f);
          float32x4_t acc2 = vdupq_n_f32(0.0f);
          float32x4_t acc3 = vdupq_n_f32(0.0f);
          float32x4_t acc4 = vdupq_n_f32(0.0f);
          float32x4_t acc5 = vdupq_n_f32(0.0f);
          float32x4_t acc6 = vdupq_n_f32(0.0f);
          float32x4_t acc7 = vdupq_n_f32(0.0f);
          float32x4_t acc8 = vdupq_n_f32(0.0f);
          float32x4_t acc9 = vdupq_n_f32(0.0f);
          float32x4_t acc10 = vdupq_n_f32(0.0f);
          float32x4_t acc11 = vdupq_n_f32(0.0f);
          float32x4_t acc12 = vdupq_n_f32(0.0f);
          float32x4_t acc13 = vdupq_n_f32(0.0f);
          float32x4_t acc14 = vdupq_n_f32(0.0f);
          float32x4_t acc15 = vdupq_n_f32(0.0f);

          const float *ptrB0 = &B_t[(l)*m];
          const float *ptrB1 = &B_t[(l + 1) * m];
          const float *ptrB2 = &B_t[(l + 2) * m];
          const float *ptrB3 = &B_t[(l + 3) * m];
          const float *ptrB4 = &B_t[(l + 4) * m];
          const float *ptrB5 = &B_t[(l + 5) * m];
          const float *ptrB6 = &B_t[(l + 6) * m];
          const float *ptrB7 = &B_t[(l + 7) * m];
          const float *ptrB8 = &B_t[(l + 8) * m];
          const float *ptrB9 = &B_t[(l + 9) * m];
          const float *ptrB10 = &B_t[(l + 10) * m];
          const float *ptrB11 = &B_t[(l + 11) * m];
          const float *ptrB12 = &B_t[(l + 12) * m];
          const float *ptrB13 = &B_t[(l + 13) * m];
          const float *ptrB14 = &B_t[(l + 14) * m];
          const float *ptrB15 = &B_t[(l + 15) * m];

          size_t j = 0;
          for (; j + 3 < m; j += 4)
          {
            float32x4_t a = vld1q_f32(ptrA + j);

            acc0 = vfmaq_f32(acc0, a, vld1q_f32(ptrB0 + j));
            acc1 = vfmaq_f32(acc1, a, vld1q_f32(ptrB1 + j));
            acc2 = vfmaq_f32(acc2, a, vld1q_f32(ptrB2 + j));
            acc3 = vfmaq_f32(acc3, a, vld1q_f32(ptrB3 + j));
            acc4 = vfmaq_f32(acc4, a, vld1q_f32(ptrB4 + j));
            acc5 = vfmaq_f32(acc5, a, vld1q_f32(ptrB5 + j));
            acc6 = vfmaq_f32(acc6, a, vld1q_f32(ptrB6 + j));
            acc7 = vfmaq_f32(acc7, a, vld1q_f32(ptrB7 + j));
            acc8 = vfmaq_f32(acc8, a, vld1q_f32(ptrB8 + j));
            acc9 = vfmaq_f32(acc9, a, vld1q_f32(ptrB9 + j));
            acc10 = vfmaq_f32(acc10, a, vld1q_f32(ptrB10 + j));
            acc11 = vfmaq_f32(acc11, a, vld1q_f32(ptrB11 + j));
            acc12 = vfmaq_f32(acc12, a, vld1q_f32(ptrB12 + j));
            acc13 = vfmaq_f32(acc13, a, vld1q_f32(ptrB13 + j));
            acc14 = vfmaq_f32(acc14, a, vld1q_f32(ptrB14 + j));
            acc15 = vfmaq_f32(acc15, a, vld1q_f32(ptrB15 + j));
          }

          float s0 = vaddvq_f32(acc0);
          float s1 = vaddvq_f32(acc1);
          float s2 = vaddvq_f32(acc2);
          float s3 = vaddvq_f32(acc3);
          float s4 = vaddvq_f32(acc4);
          float s5 = vaddvq_f32(acc5);
          float s6 = vaddvq_f32(acc6);
          float s7 = vaddvq_f32(acc7);
          float s8 = vaddvq_f32(acc8);
          float s9 = vaddvq_f32(acc9);
          float s10 = vaddvq_f32(acc10);
          float s11 = vaddvq_f32(acc11);
          float s12 = vaddvq_f32(acc12);
          float s13 = vaddvq_f32(acc13);
          float s14 = vaddvq_f32(acc14);
          float s15 = vaddvq_f32(acc15);

          for (; j < m; ++j)
          {
            float a_val = ptrA[j];
            s0 += a_val * ptrB0[j];
            s1 += a_val * ptrB1[j];
            s2 += a_val * ptrB2[j];
            s3 += a_val * ptrB3[j];
            s4 += a_val * ptrB4[j];
            s5 += a_val * ptrB5[j];
            s6 += a_val * ptrB6[j];
            s7 += a_val * ptrB7[j];
            s8 += a_val * ptrB8[j];
            s9 += a_val * ptrB9[j];
            s10 += a_val * ptrB10[j];
            s11 += a_val * ptrB11[j];
            s12 += a_val * ptrB12[j];
            s13 += a_val * ptrB13[j];
            s14 += a_val * ptrB14[j];
            s15 += a_val * ptrB15[j];
          }

          C[i * k + (l)] = s0;
          C[i * k + (l + 1)] = s1;
          C[i * k + (l + 2)] = s2;
          C[i * k + (l + 3)] = s3;
          C[i * k + (l + 4)] = s4;
          C[i * k + (l + 5)] = s5;
          C[i * k + (l + 6)] = s6;
          C[i * k + (l + 7)] = s7;
          C[i * k + (l + 8)] = s8;
          C[i * k + (l + 9)] = s9;
          C[i * k + (l + 10)] = s10;
          C[i * k + (l + 11)] = s11;
          C[i * k + (l + 12)] = s12;
          C[i * k + (l + 13)] = s13;
          C[i * k + (l + 14)] = s14;
          C[i * k + (l + 15)] = s15;
        }

        for (; l + 7 < lE;
             l += 8)
        { // Better use of A, use it 4 times before reload

          float32x4_t acc0 = vdupq_n_f32(0.0f);
          float32x4_t acc1 = vdupq_n_f32(0.0f);
          float32x4_t acc2 = vdupq_n_f32(0.0f);
          float32x4_t acc3 = vdupq_n_f32(0.0f);
          float32x4_t acc4 = vdupq_n_f32(0.0f);
          float32x4_t acc5 = vdupq_n_f32(0.0f);
          float32x4_t acc6 = vdupq_n_f32(0.0f);
          float32x4_t acc7 = vdupq_n_f32(0.0f);

          const float *ptrB0 = &B_t[l * m];
          const float *ptrB1 = &B_t[(l + 1) * m];
          const float *ptrB2 = &B_t[(l + 2) * m];
          const float *ptrB3 = &B_t[(l + 3) * m];
          const float *ptrB4 = &B_t[(l + 4) * m];
          const float *ptrB5 = &B_t[(l + 5) * m];
          const float *ptrB6 = &B_t[(l + 6) * m];
          const float *ptrB7 = &B_t[(l + 7) * m];

          size_t j = 0;
          for (; j + 3 < m; j += 4)
          {
            float32x4_t a = vld1q_f32(ptrA + j);

            acc0 = vfmaq_f32(acc0, a, vld1q_f32(ptrB0 + j));
            acc1 = vfmaq_f32(acc1, a, vld1q_f32(ptrB1 + j));
            acc2 = vfmaq_f32(acc2, a, vld1q_f32(ptrB2 + j));
            acc3 = vfmaq_f32(acc3, a, vld1q_f32(ptrB3 + j));
            acc4 = vfmaq_f32(acc4, a, vld1q_f32(ptrB4 + j));
            acc5 = vfmaq_f32(acc5, a, vld1q_f32(ptrB5 + j));
            acc6 = vfmaq_f32(acc6, a, vld1q_f32(ptrB6 + j));
            acc7 = vfmaq_f32(acc7, a, vld1q_f32(ptrB7 + j));
          }

          float s0 = vaddvq_f32(acc0);
          float s1 = vaddvq_f32(acc1);
          float s2 = vaddvq_f32(acc2);
          float s3 = vaddvq_f32(acc3);
          float s4 = vaddvq_f32(acc4);
          float s5 = vaddvq_f32(acc5);
          float s6 = vaddvq_f32(acc6);
          float s7 = vaddvq_f32(acc7);

          for (; j < m; ++j)
          {
            float a_val = ptrA[j];
            s0 += a_val * ptrB0[j];
            s1 += a_val * ptrB1[j];
            s2 += a_val * ptrB2[j];
            s3 += a_val * ptrB3[j];
            s4 += a_val * ptrB4[j];
            s5 += a_val * ptrB5[j];
            s6 += a_val * ptrB6[j];
            s7 += a_val * ptrB7[j];
          }

          C[i * k + (l)] = s0;
          C[i * k + (l + 1)] = s1;
          C[i * k + (l + 2)] = s2;
          C[i * k + (l + 3)] = s3;
          C[i * k + (l + 4)] = s4;
          C[i * k + (l + 5)] = s5;
          C[i * k + (l + 6)] = s6;
          C[i * k + (l + 7)] = s7;
        }

        for (; l + 3 < lE; l += 4)
        {
          float32x4_t acc0 = vdupq_n_f32(0.0f);
          float32x4_t acc1 = vdupq_n_f32(0.0f);
          float32x4_t acc2 = vdupq_n_f32(0.0f);
          float32x4_t acc3 = vdupq_n_f32(0.0f);

          const float *ptrB0 = &B_t[l * m];
          const float *ptrB1 = &B_t[(l + 1) * m];
          const float *ptrB2 = &B_t[(l + 2) * m];
          const float *ptrB3 = &B_t[(l + 3) * m];

          size_t j = 0;
          for (; j + 3 < m; j += 4)
          {
            float32x4_t a = vld1q_f32(ptrA + j);

            acc0 = vfmaq_f32(acc0, a, vld1q_f32(ptrB0 + j));
            acc1 = vfmaq_f32(acc1, a, vld1q_f32(ptrB1 + j));
            acc2 = vfmaq_f32(acc2, a, vld1q_f32(ptrB2 + j));
            acc3 = vfmaq_f32(acc3, a, vld1q_f32(ptrB3 + j));
          }

          float s0 = vaddvq_f32(acc0);
          float s1 = vaddvq_f32(acc1);
          float s2 = vaddvq_f32(acc2);
          float s3 = vaddvq_f32(acc3);

          for (; j < m; ++j)
          {
            float a_val = ptrA[j];
            s0 += a_val * ptrB0[j];
            s1 += a_val * ptrB1[j];
            s2 += a_val * ptrB2[j];
            s3 += a_val * ptrB3[j];
          }

          C[i * k + (l)] = s0;
          C[i * k + (l + 1)] = s1;
          C[i * k + (l + 2)] = s2;
          C[i * k + (l + 3)] = s3;
        }

        for (; l < lE; ++l)
        { // matmul_cpu handling

          float32x4_t acc = vdupq_n_f32(0.0f);

          float sum = 0;
          const float *ptrB = &B_t[l * m];

          size_t j = 0;
          for (; j + 3 < m; j += 4)
          {
            float32x4_t a = vld1q_f32(ptrA + j);
            float32x4_t b = vld1q_f32(ptrB + j);
            acc = vfmaq_f32(acc, a, b);
          }
          for (; j < m; ++j)
          {
            sum += ptrA[j] * ptrB[j];
          }

          sum += vaddvq_f32(acc);
          C[i * k + l] = sum;
        }
      }
    }
  }
  tensor_free(B_t);
}

void Tensor::matmul_cpu_unrolled_8x(const float *__restrict__ A,
                                    const float *__restrict__ B,
                                    float *__restrict__ C, int n, int m,
                                    int k) const
{
  // A = nxm
  // B = mxk
  // C = nxk
  // Stride A = m
  // Stride B = k
  // Stride C = k

  std::tuple<int, int, int> tile_data = get_matmul_tile(n, m, k);
  int T_n = std::get<0>(tile_data);
  int T_k = std::get<2>(tile_data);

  float *B_t = float_size_alloc(m * k);

  simd_transpose(B, B_t, m, k);

  for (size_t ic = 0; ic < n; ic += T_n)
  {
    size_t iE = std::min(ic + T_n, static_cast<size_t>(n));
    for (size_t lc = 0; lc < k; lc += T_k)
    {
      size_t lE = std::min(lc + T_k, static_cast<size_t>(k));
      for (size_t i = ic; i < iE; ++i)
      {
        size_t l = lc;
        const float *ptrA = &A[i * m];

        for (; l + 7 < lE;
             l += 8)
        { // Better use of A, use it 4 times before reload

          float32x4_t acc0 = vdupq_n_f32(0.0f);
          float32x4_t acc1 = vdupq_n_f32(0.0f);
          float32x4_t acc2 = vdupq_n_f32(0.0f);
          float32x4_t acc3 = vdupq_n_f32(0.0f);
          float32x4_t acc4 = vdupq_n_f32(0.0f);
          float32x4_t acc5 = vdupq_n_f32(0.0f);
          float32x4_t acc6 = vdupq_n_f32(0.0f);
          float32x4_t acc7 = vdupq_n_f32(0.0f);

          const float *ptrB0 = &B_t[l * m];
          const float *ptrB1 = &B_t[(l + 1) * m];
          const float *ptrB2 = &B_t[(l + 2) * m];
          const float *ptrB3 = &B_t[(l + 3) * m];
          const float *ptrB4 = &B_t[(l + 4) * m];
          const float *ptrB5 = &B_t[(l + 5) * m];
          const float *ptrB6 = &B_t[(l + 6) * m];
          const float *ptrB7 = &B_t[(l + 7) * m];

          size_t j = 0;
          for (; j + 3 < m; j += 4)
          {
            float32x4_t a = vld1q_f32(ptrA + j);

            acc0 = vfmaq_f32(acc0, a, vld1q_f32(ptrB0 + j));
            acc1 = vfmaq_f32(acc1, a, vld1q_f32(ptrB1 + j));
            acc2 = vfmaq_f32(acc2, a, vld1q_f32(ptrB2 + j));
            acc3 = vfmaq_f32(acc3, a, vld1q_f32(ptrB3 + j));
            acc4 = vfmaq_f32(acc4, a, vld1q_f32(ptrB4 + j));
            acc5 = vfmaq_f32(acc5, a, vld1q_f32(ptrB5 + j));
            acc6 = vfmaq_f32(acc6, a, vld1q_f32(ptrB6 + j));
            acc7 = vfmaq_f32(acc7, a, vld1q_f32(ptrB7 + j));
          }

          float s0 = vaddvq_f32(acc0);
          float s1 = vaddvq_f32(acc1);
          float s2 = vaddvq_f32(acc2);
          float s3 = vaddvq_f32(acc3);
          float s4 = vaddvq_f32(acc4);
          float s5 = vaddvq_f32(acc5);
          float s6 = vaddvq_f32(acc6);
          float s7 = vaddvq_f32(acc7);

          for (; j < m; ++j)
          {
            float a_val = ptrA[j];
            s0 += a_val * ptrB0[j];
            s1 += a_val * ptrB1[j];
            s2 += a_val * ptrB2[j];
            s3 += a_val * ptrB3[j];
            s4 += a_val * ptrB4[j];
            s5 += a_val * ptrB5[j];
            s6 += a_val * ptrB6[j];
            s7 += a_val * ptrB7[j];
          }

          C[i * k + (l)] = s0;
          C[i * k + (l + 1)] = s1;
          C[i * k + (l + 2)] = s2;
          C[i * k + (l + 3)] = s3;
          C[i * k + (l + 4)] = s4;
          C[i * k + (l + 5)] = s5;
          C[i * k + (l + 6)] = s6;
          C[i * k + (l + 7)] = s7;
        }

        for (; l + 3 < lE; l += 4)
        {
          float32x4_t acc0 = vdupq_n_f32(0.0f);
          float32x4_t acc1 = vdupq_n_f32(0.0f);
          float32x4_t acc2 = vdupq_n_f32(0.0f);
          float32x4_t acc3 = vdupq_n_f32(0.0f);

          const float *ptrB0 = &B_t[l * m];
          const float *ptrB1 = &B_t[(l + 1) * m];
          const float *ptrB2 = &B_t[(l + 2) * m];
          const float *ptrB3 = &B_t[(l + 3) * m];

          size_t j = 0;
          for (; j + 3 < m; j += 4)
          {
            float32x4_t a = vld1q_f32(ptrA + j);

            acc0 = vfmaq_f32(acc0, a, vld1q_f32(ptrB0 + j));
            acc1 = vfmaq_f32(acc1, a, vld1q_f32(ptrB1 + j));
            acc2 = vfmaq_f32(acc2, a, vld1q_f32(ptrB2 + j));
            acc3 = vfmaq_f32(acc3, a, vld1q_f32(ptrB3 + j));
          }

          float s0 = vaddvq_f32(acc0);
          float s1 = vaddvq_f32(acc1);
          float s2 = vaddvq_f32(acc2);
          float s3 = vaddvq_f32(acc3);

          for (; j < m; ++j)
          {
            float a_val = ptrA[j];
            s0 += a_val * ptrB0[j];
            s1 += a_val * ptrB1[j];
            s2 += a_val * ptrB2[j];
            s3 += a_val * ptrB3[j];
          }

          C[i * k + (l)] = s0;
          C[i * k + (l + 1)] = s1;
          C[i * k + (l + 2)] = s2;
          C[i * k + (l + 3)] = s3;
        }

        for (; l < lE; ++l)
        { // matmul_cpu handling

          float32x4_t acc = vdupq_n_f32(0.0f);

          float sum = 0;
          const float *ptrB = &B_t[l * m];

          size_t j = 0;
          for (; j + 3 < m; j += 4)
          {
            float32x4_t a = vld1q_f32(ptrA + j);
            float32x4_t b = vld1q_f32(ptrB + j);
            acc = vfmaq_f32(acc, a, b);
          }
          for (; j < m; ++j)
          {
            sum += ptrA[j] * ptrB[j];
          }

          sum += vaddvq_f32(acc);
          C[i * k + l] = sum;
        }
      }
    }
  }
  tensor_free(B_t);
}

void Tensor::matmul_cpu_unrolled_4x(const float *__restrict__ A,
                                    const float *__restrict__ B,
                                    float *__restrict__ C, int n, int m,
                                    int k) const
{
  // A = nxm
  // B = mxk
  // C = nxk
  // Stride A = m
  // Stride B = k
  // Stride C = k

  std::tuple<int, int, int> tile_data = get_matmul_tile(n, m, k);
  int T_n = std::get<0>(tile_data);
  int T_k = std::get<2>(tile_data);

  float *B_t = float_size_alloc(m * k);

  simd_transpose(B, B_t, m, k);

  for (size_t ic = 0; ic < n; ic += T_n)
  {
    size_t iE = std::min(ic + T_n, static_cast<size_t>(n));
    for (size_t lc = 0; lc < k; lc += T_k)
    {
      size_t lE = std::min(lc + T_k, static_cast<size_t>(k));
      for (size_t i = ic; i < iE; ++i)
      {
        size_t l = lc;
        const float *ptrA = &A[i * m];

        for (; l + 3 < lE;
             l += 4)
        { // Better use of A, use it 4 times before reload

          float32x4_t acc0 = vdupq_n_f32(0.0f);
          float32x4_t acc1 = vdupq_n_f32(0.0f);
          float32x4_t acc2 = vdupq_n_f32(0.0f);
          float32x4_t acc3 = vdupq_n_f32(0.0f);

          const float *ptrB0 = &B_t[l * m];
          const float *ptrB1 = &B_t[(l + 1) * m];
          const float *ptrB2 = &B_t[(l + 2) * m];
          const float *ptrB3 = &B_t[(l + 3) * m];

          size_t j = 0;
          for (; j + 3 < m; j += 4)
          {
            float32x4_t a = vld1q_f32(ptrA + j);

            acc0 = vfmaq_f32(acc0, a, vld1q_f32(ptrB0 + j));
            acc1 = vfmaq_f32(acc1, a, vld1q_f32(ptrB1 + j));
            acc2 = vfmaq_f32(acc2, a, vld1q_f32(ptrB2 + j));
            acc3 = vfmaq_f32(acc3, a, vld1q_f32(ptrB3 + j));
          }

          float s0 = vaddvq_f32(acc0);
          float s1 = vaddvq_f32(acc1);
          float s2 = vaddvq_f32(acc2);
          float s3 = vaddvq_f32(acc3);

          for (; j < m; ++j)
          {
            float a_val = ptrA[j];
            s0 += a_val * ptrB0[j];
            s1 += a_val * ptrB1[j];
            s2 += a_val * ptrB2[j];
            s3 += a_val * ptrB3[j];
          }

          C[i * k + (l)] = s0;
          C[i * k + (l + 1)] = s1;
          C[i * k + (l + 2)] = s2;
          C[i * k + (l + 3)] = s3;
        }

        for (; l < lE; ++l)
        { // matmul_cpu handling

          float32x4_t acc = vdupq_n_f32(0.0f);

          float sum = 0;
          const float *ptrB = &B_t[l * m];

          size_t j = 0;
          for (; j + 3 < m; j += 4)
          {
            float32x4_t a = vld1q_f32(ptrA + j);
            float32x4_t b = vld1q_f32(ptrB + j);
            acc = vfmaq_f32(acc, a, b);
          }
          for (; j < m; ++j)
          {
            sum += ptrA[j] * ptrB[j];
          }

          sum += vaddvq_f32(acc);
          C[i * k + l] = sum;
        }
      }
    }
  }
  tensor_free(B_t);
}

// Main issue 1: Not targeting performance cores, learned from profiler results
// and apple silicion optimization pdf (v4) Main issue 2: Assumption of square
// matricies: but in neural network matreces are rarely square (batch size
// usually less or more than dimension) Issue 3: Should focus on tiling B into
// cache (leaving some space for A), since it needs to be accesed multiple times
// for one val/row of A.
//  Result: Updated tile code to be rectangular and weight on T_k, ignore m
//  dimension. (Also made L1 tiling much more conservative)
void Tensor::matmul_cpu_tiled_old(const float *__restrict__ A,
                                  const float *__restrict__ B,
                                  float *__restrict__ C, int n, int m,
                                  int k) const
{
  constexpr size_t L1_bytes = 64 * 1024;
  constexpr size_t L2_bytes = 4 * 1024 * 1024;
  constexpr int cache_line_size = 128;

  // Choose between L1 and L2 cache based on matrix size
  size_t cache_line_floats = cache_line_size / sizeof(float);
  size_t usable_cache_bytes;
  if ((n * m + m * k + n * k) * sizeof(float) <= L1_bytes)
  {
    // Only use max 2/3 of the available bytes in a given cache.
    usable_cache_bytes = L1_bytes / 1.5;
  }
  else
  {
    usable_cache_bytes = L2_bytes / 1.5;
  }

  size_t usable_cache_floats = usable_cache_bytes / sizeof(float);

  int mat_tile = static_cast<int>(sqrt(usable_cache_floats / 3));

  mat_tile -= mat_tile % cache_line_floats;
  if (mat_tile == 0)
  {
    mat_tile = cache_line_floats;
  }

  for (size_t ic = 0; ic < n; ic += mat_tile)
  {
    size_t iE = std::min(ic + mat_tile, static_cast<size_t>(n));

    for (size_t lc = 0; lc < k; lc += mat_tile)
    {
      size_t lE = std::min(lc + mat_tile, static_cast<size_t>(k));

      for (size_t i = ic; i < iE; ++i)
      {
        const float *ptrA = &A[i * m];
        for (size_t l = lc; l < lE; ++l)
        {
          float sum = 0;
          const float *ptrB = &B[l];

          for (size_t j = 0; j < m; ++j)
          {
            sum += ptrA[j] * ptrB[j * k];
          }

          C[i * k + l] = sum;
        }
      }
    }
  }
}

void Tensor::matmul_cpu_tiled(const float *__restrict__ A,
                              const float *__restrict__ B,
                              float *__restrict__ C, int n, int m,
                              int k) const
{
  // A = nxm
  // B = mxk
  // C = nxk
  // Stride A = m
  // Stride B = k
  // Stride C = k

  std::tuple<int, int, int> tile_data = get_matmul_tile(n, m, k);
  int T_n = std::get<0>(tile_data);
  int T_k = std::get<2>(tile_data);

  for (size_t ic = 0; ic < n; ic += T_n)
  {
    size_t iE = std::min(ic + T_n, static_cast<size_t>(n));

    for (size_t lc = 0; lc < k; lc += T_k)
    {
      size_t lE = std::min(lc + T_k, static_cast<size_t>(k));

      for (size_t i = ic; i < iE; ++i)
      {
        const float *ptrA = &A[i * m];
        for (size_t l = lc; l < lE; ++l)
        {
          float sum = 0;
          const float *ptrB = &B[l];

          for (size_t j = 0; j < m; ++j)
          {
            sum += ptrA[j] * ptrB[j * k];
          }

          C[i * k + l] = sum;
        }
      }
    }
  }
}

// void Tensor::matmul_metal(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int n, int m, int k) const {
//
// }

#ifndef WINDOWS_VER
void Tensor::matmul_amx(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int n, int m, int k) const
{

  // TODO: compute bitmasks once at start.
  // TODO: Use extra registers for pipelining/prefetching

  const size_t TILE = 32; // 2x2 16x16 tiles.
  float *A_t = float_size_alloc(n * m);
  simd_transpose(A, A_t, n, m);

  float *tmp = float_size_alloc(32 * 32);

  AMX_SET();
  for (size_t ic = 0; ic < n; ic += TILE)
  {
    size_t valid_a = std::min(TILE, n - ic);
    for (size_t lc = 0; lc < k; lc += TILE)
    {
      size_t valid_b = std::min(TILE, k - lc);
      float *ptrC = &C[ic * k + lc];

      for (size_t j = 0; j < m; ++j)
      {
        const float *ptrA = &A_t[j * n + ic];
        const float *ptrB = &B[j * k + lc];
        amx_kernel_f32(ptrA, ptrB, valid_a, valid_b, j == 0);
      }

      store_amx_output(tmp);
      

      for (size_t r = 0; r < valid_a; r++)
      {
        for (size_t c = 0; c < valid_b; c++)
        {
          ptrC[r * k + c] = tmp[c * 32 + r];
        }
      }
    }
  }
  AMX_CLR();

  tensor_free(A_t);
  tensor_free(tmp);
}
#endif

void Tensor::matmul_cpu(const float *__restrict__ A,
                        const float *__restrict__ B, float *__restrict__ C,
                        int n, int m, int k) const
{
  // A = nxm
  // B = mxk
  // C = nxk
  // Stride A = m
  // Stride B = k
  // Stride C = k

  std::tuple<int, int, int> tile_data = get_matmul_tile(n, m, k);
  int T_n = std::get<0>(tile_data);
  int T_k = std::get<2>(tile_data);

  float *B_t = float_size_alloc(m * k);

  simd_transpose(B, B_t, m, k);

  for (size_t ic = 0; ic < n; ic += T_n)
  {
    size_t iE = std::min(ic + T_n, static_cast<size_t>(n));
    for (size_t lc = 0; lc < k; lc += T_k)
    {
      size_t lE = std::min(lc + T_k, static_cast<size_t>(k));
      for (size_t i = ic; i < iE; ++i)
      {
        const float *ptrA = &A[i * m];
        for (size_t l = lc; l < lE; ++l)
        {
          float32x4_t acc = vdupq_n_f32(0.0f);

          float sum = 0;
          const float *ptrB = &B_t[l * m];

          size_t j = 0;
          for (; j + 3 < m; j += 4)
          {
            float32x4_t a = vld1q_f32(ptrA + j);
            float32x4_t b = vld1q_f32(ptrB + j);
            acc = vfmaq_f32(acc, a, b);
          }
          for (; j < m; ++j)
          {
            sum += ptrA[j] * ptrB[j];
          }

          sum += vaddvq_f32(acc);
          C[i * k + l] = sum;
        }
      }
    }
  }
  tensor_free(B_t);
}

void Tensor::matmul_cpu_naive(const float *__restrict__ A,
                              const float *__restrict__ B,
                              float *__restrict__ C, int n, int m,
                              int k) const
{
  for (size_t i = 0; i < n; ++i)
  {
    for (size_t l = 0; l < k; ++l)
    {
      float sum = 0;
      for (size_t j = 0; j < m; ++j)
      {
        sum += A[i * m + j] * B[j * k + l];
      }
      C[i * k + l] = sum;
    }
  }
}

void Tensor::simd_transpose(const float *A, float *B, int n, int m, int z,
                            const int *dists_new) const
{
  // We choose to repeat code rather than make a temp dists var so that the
  // original dists_new can stay const (cannot free if we use const
  // temp_dists_new).

  if (!dists_new)
  {
    size_t offset = n * m * z;
    // Tile for same reasons as matmul (minimize cache misses)
    for (size_t ic = 0; ic + tile <= n; ic += tile)
    {
      for (size_t jc = 0; jc + tile <= m; jc += tile)
      {
        for (size_t i = ic; i < ic + tile; i += 4)
        {
          for (size_t j = jc; j < jc + tile; j += 4)
          {
            // Does 4x4 sections, and a scalar cleanup.
            // Load 16 elements from A to tranpose into B
            //  a = [a0 a1 a2 a3]
            //  b = [b0 b1 b2 b3]
            //  c = [c0 c1 c2 c3]
            //  d = [d0 d1 d2 d3]
            float32x4_t a = vld1q_f32(&A[offset + i * m + j]);
            float32x4_t b = vld1q_f32(&A[offset + (i + 1) * m + j]);
            float32x4_t c = vld1q_f32(&A[offset + (i + 2) * m + j]);
            float32x4_t d = vld1q_f32(&A[offset + (i + 3) * m + j]);

            // Transpose halves (swap even and odd lanes)
            //[a0 a1 a2 a3]      [a0 b0 a2 b2]
            //[b0 b1 b2 b3]  →   [a1 b1 a3 b3]
            float32x4x2_t p0 = vtrnq_f32(a, b);
            float32x4x2_t p1 = vtrnq_f32(c, d);

            // Combine halves

            // low(p0[0]) = [a0 b0]
            // low(p1[0]) = [c0 d0]
            // → r0 = [a0 b0 c0 d0]
            float32x4_t r0 =
                vcombine_f32(vget_low_f32(p0.val[0]), vget_low_f32(p1.val[0]));
            float32x4_t r1 =
                vcombine_f32(vget_low_f32(p0.val[1]), vget_low_f32(p1.val[1]));

            // high(p0[0]) = [a2 b2]
            // high(p1[0]) = [c2 d2]
            // → r2 = [a2 b2 c2 d2]
            float32x4_t r2 = vcombine_f32(vget_high_f32(p0.val[0]),
                                          vget_high_f32(p1.val[0]));
            float32x4_t r3 = vcombine_f32(vget_high_f32(p0.val[1]),
                                          vget_high_f32(p1.val[1]));

            // Store into B
            vst1q_f32(&B[j * n + i], r0);
            vst1q_f32(&B[(j + 1) * n + i], r1);
            vst1q_f32(&B[(j + 2) * n + i], r2);
            vst1q_f32(&B[(j + 3) * n + i], r3);
          }
        }
      }
    }

    // Scalar Clean up what was missed by tiling

    // Handles leftover rows - Bottom Rectangle
    for (size_t i = n - (n % tile); i < n; ++i)
    {
      for (size_t j = 0; j < m; ++j)
      {
        B[j * n + i] = A[offset + i * m + j];
      }
    }

    // Corner:
    // i >= n-(n%tile)
    // j >= m-(m%tile)

    // Handle leftover colouns (ignoring final few rows overlapping with above
    // loop) Basically the top right rectangle
    for (size_t i = 0; i < n - (n % tile); ++i)
    {
      for (size_t j = m - (m % tile); j < m; ++j)
      {
        B[j * n + i] = A[offset + i * m + j];
      }
    }
  }
  else
  {
    for (size_t ic = 0; ic + tile <= n; ic += tile)
    {
      for (size_t jc = 0; jc + tile <= m; jc += tile)
      {
        for (size_t i = ic; i < ic + tile; i += 4)
        {
          for (size_t j = jc; j < jc + tile; j += 4)
          {
            float32x4_t a = vld1q_f32(
                &A[z * dists_new[0] + i * dists_new[1] + j * dists_new[2]]);
            float32x4_t b =
                vld1q_f32(&A[z * dists_new[0] + (i + 1) * dists_new[1] +
                             j * dists_new[2]]);
            float32x4_t c =
                vld1q_f32(&A[z * dists_new[0] + (i + 2) * dists_new[1] +
                             j * dists_new[2]]);
            float32x4_t d =
                vld1q_f32(&A[z * dists_new[0] + (i + 3) * dists_new[1] +
                             j * dists_new[2]]);

            float32x4x2_t p0 = vtrnq_f32(a, b);
            float32x4x2_t p1 = vtrnq_f32(c, d);

            float32x4_t r0 =
                vcombine_f32(vget_low_f32(p0.val[0]), vget_low_f32(p1.val[0]));
            float32x4_t r1 =
                vcombine_f32(vget_low_f32(p0.val[1]), vget_low_f32(p1.val[1]));
            float32x4_t r2 = vcombine_f32(vget_high_f32(p0.val[0]),
                                          vget_high_f32(p1.val[0]));
            float32x4_t r3 = vcombine_f32(vget_high_f32(p0.val[1]),
                                          vget_high_f32(p1.val[1]));

            vst1q_f32(&B[j * n + i], r0);
            vst1q_f32(&B[(j + 1) * n + i], r1);
            vst1q_f32(&B[(j + 2) * n + i], r2);
            vst1q_f32(&B[(j + 3) * n + i], r3);
          }
        }
      }
    }

    for (size_t i = n - (n % tile); i < n; ++i)
    {
      for (size_t j = 0; j < m; ++j)
      {
        B[j * n + i] =
            A[z * dists_new[0] + i * dists_new[1] + j * dists_new[2]];
      }
    }

    for (size_t i = 0; i < n - (n % tile); ++i)
    {
      for (size_t j = m - (m % tile); j < m; ++j)
      {
        B[j * n + i] =
            A[z * dists_new[0] + i * dists_new[1] + j * dists_new[2]];
      }
    }
  }
}

Tensor Tensor::matmul(const Tensor &other) const
{
  if (other.get_dim_len() == 1 && dim_len == 1)
  {
    // Dimension 1 x 1 = Dot product

    if (other.get_dims_index(0) == dims[0])
    {
      int *new_dims = int_size_alloc(1);
      new_dims[0] = 1;
      float data_out = 0;
      float32x4_t acc = vdupq_n_f32(0.0f);

      // SIMD loop just loops through both vectors and accumulates their dot
      // product sum into one SIMD load, doing the final addition across the 4
      // lanes at the end. Makes use of basic, heuristic tiling to try and keep
      // very large vectors within L1 cache, but this dosent really affect usual
      // small scale performance
      for (size_t jc = 0; jc < data_len; jc += tile)
      {
        size_t jE = std::min(jc + tile, static_cast<size_t>(data_len));
        const float *oPtr = other.data + jc;
        float *tPtr = data + jc;
        for (size_t j = jc; j + 3 < jE; j += 4)
        {
          float32x4_t a = vld1q_f32(oPtr);
          float32x4_t b = vld1q_f32(tPtr);
          oPtr += 4;
          tPtr += 4;
          acc = vfmaq_f32(acc, a, b);
        }
        for (size_t j = 0; j < (jE % 4); ++j)
        {
          data_out += (*oPtr) * (*tPtr);
          oPtr += 1;
          tPtr += 1;
        }
      }
      data_out += vaddvq_f32(acc);
      Tensor ret = Tensor(new_dims, 1, data_out);
      free(new_dims);
      return ret;
    }
    throw std::invalid_argument("Invalid dot product dimensions!");
  }
  else if (other.get_dim_len() == 1 && dim_len == 2)
  {
    // dimension 2 x 1 = Vector Product

    // n x m X m x 1 = n x 1
    if (other.get_dims_index(0) == dims[1])
    {
      int *new_dims = int_size_alloc(1);
      new_dims[0] = dims[0];

      float *data_out = float_size_alloc(new_dims[0]);

      // Loop through all rows, accumulating dot product with vector. After
      // computing each dot product row_n * vector, add it to corresponding row
      // index of output array. Dot product computation is sped up with SIMD,
      // same as above.
      for (size_t ic = 0; ic < new_dims[0]; ic += tile)
      {
        size_t iE = std::min(ic + tile, static_cast<size_t>(new_dims[0]));
        for (size_t i = ic; i < iE; ++i)
        {
          float sum = 0;
          float32x4_t acc = vdupq_n_f32(0.0f);
          for (size_t jc = 0; jc < dims[1]; jc += tile)
          {
            size_t jE = std::min(jc + tile, static_cast<size_t>(dims[1]));
            float *ptrA = &data[i * dims[1] + jc];
            float *ptrB = &(other.data[jc]);
            for (size_t j = jc; j + 3 < jE; j += 4)
            {
              float32x4_t a = vld1q_f32(ptrA);
              float32x4_t b = vld1q_f32(ptrB);
              ptrA += 4;
              ptrB += 4;
              acc = vfmaq_f32(acc, a, b);
            }
            for (size_t j = 0; j < (jE % 4); ++j)
            {
              sum += (*ptrA) * (*ptrB);
              ptrA += 1;
              ptrB += 1;
            }
          }
          sum += vaddvq_f32(acc);
          data_out[i] = sum;
        }
      }

      Tensor ret = Tensor(new_dims, 1, data_out, false);
      return ret;
    }
    throw std::invalid_argument("Invalid matrix-vector product dimensions!");
  }
  else if (other.get_dim_len() == 2 && dim_len == 1)
  {
    // dimension 1 x 2 = Vector Product
    // 1 x m X m x k = 1 x k
    if (other.get_dims_index(0) == dims[0])
    {
      int *new_dims = int_size_alloc(1);

      // Transpose to avoid col-wide inefficent access

      size_t m = other.get_dims_index(0);
      size_t k = other.get_dims_index(1);

      float *other_t = float_size_alloc(m * k);

      float *data_other = other.data;
      simd_transpose(data_other, other_t, m, k);

      new_dims[0] = k;

      float *data_out = float_size_alloc(new_dims[0]);

      // After transposing, the workings of this function is exactly the same as
      // the matrix-vector product.
      for (size_t ic = 0; ic < new_dims[0]; ic += tile)
      {
        size_t iE = std::min(ic + tile, static_cast<size_t>(new_dims[0]));
        for (size_t i = ic; i < iE; ++i)
        {
          float sum = 0;
          float32x4_t acc = vdupq_n_f32(0.0f);
          for (size_t jc = 0; jc < dims[0]; jc += tile)
          {
            size_t jE = std::min(jc + tile, static_cast<size_t>(dims[0]));
            float *ptrA = &data[jc];
            float *ptrB = &other_t[i * dims[0] + jc];
            for (size_t j = jc; j + 3 < jE; j += 4)
            {
              float32x4_t a = vld1q_f32(ptrA);
              float32x4_t b = vld1q_f32(ptrB);
              ptrA += 4;
              ptrB += 4;
              acc = vfmaq_f32(acc, a, b);
            }
            for (size_t j = 0; j < (jE % 4); ++j)
            {
              sum += (*ptrA) * (*ptrB);
              ptrA += 1;
              ptrB += 1;
            }
          }
          sum += vaddvq_f32(acc);
          data_out[i] = sum;
        }
      }
      tensor_free(other_t);
      Tensor ret = Tensor(new_dims, 1, data_out, false);
      return ret;
    }
    throw std::invalid_argument("Invalid vector-matrix product dimensions!");
  }
  else if (other.get_dim_len() == 2 && dim_len == 2)
  {
    // Dimension 2 x 2 = Tensor multiplication
    // Will perform This X Other
    if (dims[1] == other.get_dims_index(0))
    {
      int *new_dims = int_size_alloc(2);

      new_dims[0] = dims[0];
      new_dims[1] = other.get_dims_index(1);

      float *data_out = float_size_alloc(new_dims[0] * new_dims[1]);

      // Switch between different matmul alg's for profiling.
      // matmul_cuda(data, other.data, data_out, new_dims[0], dims[1], new_dims[1]);
      matmul_amx(data, other.data, data_out, new_dims[0], dims[1], new_dims[1]); // MAC ONLY
      // matmul_cpu_unrolled_16x(data, other.data, data_out, new_dims[0], dims[1], new_dims[1]);
      // matmul_cpu_unrolled_8x(data, other.data, data_out, new_dims[0], dims[1], new_dims[1]);
      // matmul_cpu_unrolled_4x(data, other.data, data_out, new_dims[0],dims[1], new_dims[1]);
      // matmul_cpu(data, other.data, data_out,new_dims[0], dims[1], new_dims[1]);
      // matmul_cpu_outer(data, other.data, data_out, new_dims[0], dims[1], new_dims[1]);
      // matmul_cpu_tiled(data, other.data, data_out, new_dims[0], dims[1], new_dims[1]);
      // matmul_cpu_tiled_old(data, other.data, data_out, new_dims[0], dims[1], new_dims[1]);
      // matmul_cpu_naive(data, other.data, data_out, new_dims[0], dims[1], new_dims[1]);
      Tensor ret = Tensor(new_dims, 2, data_out, false);
      return ret;
    }
    throw std::invalid_argument("Invalid matrix-matrix product dimensions!");
  }
  else if (other.get_dim_len() >= 2 && dim_len >= 2)
  {
    // Dimension n x n = Batched matrix multiplaction with broadcasting
    // Will perform This X Other, batched
    int other_dim_len = other.get_dim_len();
    if (dims[dim_len - 1] == other.get_dims_index(other_dim_len - 2))
    {
      int broadcast_dim_len =
          std::max(static_cast<int>(dim_len), other_dim_len);
      int *broadcast_dims = int_size_alloc(broadcast_dim_len);

      // Select which dimension will be broadcasted - shorter one will be
      // broadcasted.
      if (dim_len >= other_dim_len)
      {
        int diff = broadcast_dim_len - other_dim_len;
        for (int i = dim_len - 3; i >= 0;
             --i)
        { // Loop through all dimensions except last 2 (which are
          // reserved for matmul)
          int i_other = i - diff;
          if (i_other < 0)
          { // If dimension is shorter, replace all unfilled
            // dimensions with longer dimensions.
            broadcast_dims[i] = dims[i];
          }
          else
          {
            int other_dim = other.get_dims_index(i_other);
            if (dims[i] == other_dim ||
                other_dim ==
                    1)
            { // If dimension size is the same or 1, set to main
              // dimension (1 case will be handled by strides)
              broadcast_dims[i] = dims[i];
            }
            else if (dims[i] == 1)
            {
              broadcast_dims[i] =
                  other_dim; // Same goes for the other set of dimensions.
            }
            else
            { // Dimensions incompatible (not the same or atleast one is
              // not 1)
              free(broadcast_dims);
              throw std::invalid_argument(
                  "Incompatible dimensions for matmul batch broadcasting!");
            }
          }
        }
      }
      else
      {
        // Same code if other dimension is shorter.
        int diff = broadcast_dim_len - dim_len;
        for (int i = other_dim_len - 3; i >= 0; --i)
        {
          int other_dim = other.get_dims_index(i);
          int i_this = i - diff;
          if (i_this < 0)
          {
            broadcast_dims[i] = other_dim;
          }
          else
          {
            if (dims[i_this] == other_dim || other_dim == 1)
            {
              broadcast_dims[i] = dims[i_this];
            }
            else if (dims[i_this] == 1)
            {
              broadcast_dims[i] = other_dim;
            }
            else
            {
              free(broadcast_dims);
              throw std::invalid_argument(
                  "Incompatible dimensions for matmul batch broadcasting!");
            }
          }
        }
      }

      // We MUST allocate this on the heap due to how the destructor for this
      // class works - free will fail.
      //  - Could add a boolean flag to notify destructor if stack-allocated,
      //  but increased complexity/less readable
      int *bmm_shape = int_size_alloc(3);

      bmm_shape[0] = 1;
      bmm_shape[1] = dims[dim_len - 2];
      bmm_shape[2] = other.get_dims_index(other_dim_len - 1);
      for (size_t i = 0; i < broadcast_dim_len - 2; ++i)
      {
        bmm_shape[0] *= broadcast_dims[i];
      }

      // Broadcast: preserve last two dimensions for matmul.
      broadcast_dims[broadcast_dim_len - 2] = dims[dim_len - 2];
      broadcast_dims[broadcast_dim_len - 1] = dims[dim_len - 1];
      int *this_dists =
          get_broadcasted_strides(broadcast_dims, broadcast_dim_len);

      broadcast_dims[broadcast_dim_len - 2] =
          other.get_dims_index(other_dim_len - 2);
      broadcast_dims[broadcast_dim_len - 1] =
          other.get_dims_index(other_dim_len - 1);
      int *other_dists =
          other.get_broadcasted_strides(broadcast_dims, broadcast_dim_len);

      free(broadcast_dims);

      int n_threads = std::thread::hardware_concurrency();

      // Avoid malloc to call constructor
      std::thread *threads = new std::thread[n_threads];

      float *data_out =
          float_size_alloc(bmm_shape[0] * dims[dim_len - 2] * bmm_shape[2]);

      // Each thread handles an set of indivisual slices (divided evenly between
      // all possible threads)
      for (size_t t = 0; t < n_threads; ++t)
      {
        threads[t] = std::thread([&, t]()
                                 {
          for (size_t i = t; i < bmm_shape[0]; i += n_threads) {
            matmul_cpu_batched(data, other.data, data_out, this_dists,
                               other_dists, dims[dim_len - 2],
                               dims[dim_len - 1], bmm_shape[2], i);
          } });
      }

      // Wait for all threads to finish.
      for (size_t i = 0; i < n_threads; ++i)
      {
        threads[i].join();
      }

      delete[] threads;

      free(this_dists);
      free(other_dists);

      Tensor ret = Tensor(bmm_shape, 3, data_out, false);
      return ret;
    }
    else
    {
      throw std::invalid_argument(
          "Invalid batched matrix-matrix product dimensions!");
    }
  }
  return Tensor(nullptr, 0, nullptr, false);
}

Tensor Tensor::clone() const
{
  return Tensor(dims, dim_len, data, data_len, dists);
}

Tensor Tensor::scmul(float s) const
{
  float *data_out = default_data_alloc();

  float32x4_t scalar = vdupq_n_f32(s);
  float *tPtr = data;
  float *outPtr = data_out;

  for (size_t i = 0; i + 3 < data_len; i += 4)
  {
    float32x4_t t = vld1q_f32(tPtr);
    float32x4_t res = vmulq_f32(t, scalar);
    vst1q_f32(outPtr, res);
    tPtr += 4;
    outPtr += 4;
  }

  for (size_t i = 0; i < data_len % 4; ++i)
  {
    outPtr[i] = tPtr[i] * s;
  }

  Tensor ret = Tensor(get_dims_clone(), dim_len, data_out, false);
  return ret;
}

Tensor Tensor::emul(const Tensor &other) const
{
  // SIMD largely same as add - check add method for notes.
  // Tensor - Vector element mul - specific, quicker kernel for ANN
  if (other.get_dim_len() == 1 && dim_len == 2)
  {
    if (other.get_dims_index(0) == dims[1])
    {
      float *data_out = default_data_alloc();

      float *tPtr = data;
      float *outPtr = data_out;
      float *oPtr = other.data;

      for (size_t i = 0; i < dims[0]; ++i)
      {
        for (size_t j = 0; j + 3 < dims[1]; j += 4)
        {
          float32x4_t tf = vld1q_f32(tPtr);
          float32x4_t af = vld1q_f32(&oPtr[j]);
          float32x4_t mul = vmulq_f32(af, tf);
          vst1q_f32(outPtr, mul);
          tPtr += 4;
          outPtr += 4;
        }
        for (size_t j = 0; j < (dims[1] % 4); ++j)
        {
          (*outPtr) = (*tPtr) * oPtr[j];
          outPtr += 1;
          tPtr += 1;
        }
      }

      Tensor ret = Tensor(get_dims_clone(), dim_len, data_out, false);
      return ret;
    }
    else if (other.get_dims_index(0) == dims[0])
    {
      float *data_out = default_data_alloc();

      float *tPtr = data;
      float *outPtr = data_out;
      float *oPtr = other.data;

      for (size_t i = 0; i < dims[0]; ++i)
      {
        float vec_float = *oPtr;
        float32x4_t af = vdupq_n_f32(vec_float);
        for (size_t j = 0; j + 3 < dims[1]; j += 4)
        {
          float32x4_t tf = vld1q_f32(tPtr);
          float32x4_t mul = vmulq_f32(af, tf);
          vst1q_f32(outPtr, mul);
          tPtr += 4;
          outPtr += 4;
        }
        for (size_t j = 0; j < (dims[1] % 4); ++j)
        {
          (*outPtr) = (*tPtr) * vec_float;
          tPtr += 1;
          outPtr += 1;
        }
        oPtr += 1;
      }

      Tensor ret = Tensor(get_dims_clone(), dim_len, data_out, false);
      return ret;
    }
    else
    {
      throw std::invalid_argument("Invalid matrix-vector add dimensions!");
    }
  }
  else
  { // General

    for (size_t i = 0; i < dim_len; ++i)
    {
      if (dims[i] != other.get_dims_index(i))
      {
        throw std::invalid_argument("Invalid matrix-matrix add dimensions!");
      }
    }

    float *data_out = default_data_alloc();

    const float *oPtr = other.data;
    float *tPtr = data;
    float *outPtr = data_out;

    for (size_t i = 0; i + 3 < data_len; i += 4)
    {
      float32x4_t af = vld1q_f32(oPtr);
      float32x4_t tf = vld1q_f32(tPtr);
      float32x4_t mul = vmulq_f32(af, tf);
      vst1q_f32(outPtr, mul);
      oPtr += 4;
      tPtr += 4;
      outPtr += 4;
    }

    for (size_t i = 0; i < data_len % 4; ++i)
    {
      outPtr[i] = tPtr[i] * oPtr[i];
    }

    Tensor ret = Tensor(get_dims_clone(), dim_len, data_out, false);
    return ret;
  }
}

Tensor Tensor::add(const Tensor &other) const
{
  // Tensor - Vector add - specific, quicker kernel for ANN
  // Could generalize in future by doing broadcasting, but would be largely the
  // same as matmul Must be 2D and row major (cannot be semantically
  // reshaped/broadcast/transpose)
  if (other.get_dim_len() == 1 && dim_len == 2)
  {
    if (other.get_dims_index(0) == dims[1])
    { // Add 1d vector to rows

      float *data_out = default_data_alloc();

      float *tPtr = data;
      float *outPtr = data_out;
      float *oPtr = other.data;

      // SIMD Loop: will iterate over rows (contigous) and add vector, with a
      // scalar cleanup
      for (size_t i = 0; i < dims[0]; ++i)
      {
        for (size_t j = 0; j + 3 < dims[1]; j += 4)
        {
          float32x4_t tf = vld1q_f32(tPtr);
          float32x4_t af = vld1q_f32(&oPtr[j]);
          float32x4_t add = vaddq_f32(af, tf);
          vst1q_f32(outPtr, add);
          tPtr += 4;
          outPtr += 4;
        }
        for (size_t j = 0; j < (dims[1] % 4); ++j)
        {
          (*outPtr) = (*tPtr) + oPtr[j];
          outPtr += 1;
          tPtr += 1;
        }
      }

      Tensor ret = Tensor(get_dims_clone(), dim_len, data_out, false);
      return ret;
    }
    else if (other.get_dims_index(0) == dims[0])
    { // Add 1d vector to cols

      float *data_out = default_data_alloc();

      float *tPtr = data;
      float *outPtr = data_out;
      float *oPtr = other.data;

      // Since cols are non contigous in row-major memory, loop makes use that
      // each row will have the same float added to it. Loop through rows,
      // duplicating the corresponding float across one SIMD load and adding it.
      // Scalar cleanup in end Allows for the method to utilize SIMD operations
      // even for non-contigous memory.
      for (size_t i = 0; i < dims[0]; ++i)
      {
        float vec_float = *oPtr;
        float32x4_t af = vdupq_n_f32(vec_float);
        for (size_t j = 0; j + 3 < dims[1]; j += 4)
        {
          float32x4_t tf = vld1q_f32(tPtr);
          float32x4_t add = vaddq_f32(af, tf);
          vst1q_f32(outPtr, add);
          tPtr += 4;
          outPtr += 4;
        }
        for (size_t j = 0; j < (dims[1] % 4); ++j)
        {
          (*outPtr) = (*tPtr) + vec_float;
          tPtr += 1;
          outPtr += 1;
        }
        oPtr += 1;
      }

      Tensor ret = Tensor(get_dims_clone(), dim_len, data_out, false);
      return ret;
    }
    else
    {
      throw std::invalid_argument("Invalid matrix-vector add dimensions!");
    }
  }
  else
  { // General Tensor Add

    for (size_t i = 0; i < dim_len; ++i)
    {
      if (dims[i] != other.get_dims_index(i))
      {
        throw std::invalid_argument("Invalid matrix-matrix add dimensions!");
      }
    }

    float *data_out = default_data_alloc();

    const float *oPtr = other.data;
    float *tPtr = data;
    float *outPtr = data_out;

    // Very simple SIMD loop, just loop over internal data array (row-major) and
    // add.
    for (size_t i = 0; i + 3 < data_len; i += 4)
    {
      float32x4_t af = vld1q_f32(oPtr);
      float32x4_t tf = vld1q_f32(tPtr);
      float32x4_t add = vaddq_f32(af, tf);
      vst1q_f32(outPtr, add);
      oPtr += 4;
      tPtr += 4;
      outPtr += 4;
    }

    for (size_t i = 0; i < data_len % 4; ++i)
    {
      outPtr[i] = tPtr[i] + oPtr[i];
    }

    Tensor ret = Tensor(get_dims_clone(), dim_len, data_out, false);
    return ret;
  }
}

Tensor Tensor::apply(float (*func)(float)) const
{
  float *data_out = default_data_alloc();

  for (size_t i = 0; i < data_len; ++i)
  {
    data_out[i] = func(data[i]);
  }

  Tensor ret = Tensor(get_dims_clone(), dim_len, data_out, false);
  return ret;
}

Tensor Tensor::transpose2d() const
{
  if (dim_len == 1)
  {
    return clone();
  }
  else if (dim_len == 2)
  {
    float *data_out = default_data_alloc();

    simd_transpose(data, data_out, dims[0], dims[1]);

    int *dims_new = int_size_alloc(2);

    dims_new[0] = dims[1];
    dims_new[1] = dims[0];

    Tensor ret = Tensor(dims_new, dim_len, data_out, false);
    return ret;
  }
  else
  {
    throw std::invalid_argument("Invalid matrix dimensions! Must be 1D or 2D!");
  }
}

void Tensor::scmul_inplace(float s)
{
  float32x4_t scalar = vdupq_n_f32(s);
  float *tPtr = data;

  for (size_t i = 0; i + 3 < data_len; i += 4)
  {
    float32x4_t t = vld1q_f32(tPtr);
    float32x4_t res = vmulq_f32(t, scalar);
    vst1q_f32(tPtr, res);
    tPtr += 4;
  }

  for (size_t i = 0; i < data_len % 4; ++i)
  {
    tPtr[i] = tPtr[i] * s;
  }
}

void Tensor::emul_inplace(const Tensor &other)
{
  if (other.get_dim_len() == 1 && dim_len == 2)
  {
    if (other.get_dims_index(0) == dims[1])
    {
      float *tPtr = data;
      float *oPtr = other.data;

      for (size_t i = 0; i < dims[0]; ++i)
      {
        for (size_t j = 0; j + 3 < dims[1]; j += 4)
        {
          float32x4_t tf = vld1q_f32(tPtr);
          float32x4_t af = vld1q_f32(&oPtr[j]);
          float32x4_t mul = vmulq_f32(af, tf);
          vst1q_f32(tPtr, mul);
          tPtr += 4;
        }
        for (size_t j = 0; j < (dims[1] % 4); ++j)
        {
          (*tPtr) = (*tPtr) * oPtr[j];
          tPtr += 1;
        }
      }
    }
    else if (other.get_dims_index(0) == dims[0])
    {
      float *tPtr = data;
      float *oPtr = other.data;

      for (size_t i = 0; i < dims[0]; ++i)
      {
        float vec_float = *oPtr;
        float32x4_t af = vdupq_n_f32(vec_float);
        for (size_t j = 0; j + 3 < dims[1]; j += 4)
        {
          float32x4_t tf = vld1q_f32(tPtr);
          float32x4_t mul = vmulq_f32(af, tf);
          vst1q_f32(tPtr, mul);
          tPtr += 4;
        }
        for (size_t j = 0; j < (dims[1] % 4); ++j)
        {
          (*tPtr) = (*tPtr) * vec_float;
          tPtr += 1;
        }
        oPtr += 1;
      }
    }
    else
    {
      throw std::invalid_argument("Invalid matrix-vector add dimensions!");
    }
  }
  else
  {
    for (size_t i = 0; i < dim_len; ++i)
    {
      if (dims[i] != other.get_dims_index(i))
      {
        throw std::invalid_argument("Invalid matrix dimensions!");
      }
    }

    const float *oPtr = other.data;
    float *tPtr = data;

    for (size_t i = 0; i + 3 < data_len; i += 4)
    {
      float32x4_t af = vld1q_f32(oPtr);
      float32x4_t tf = vld1q_f32(tPtr);
      float32x4_t mul = vmulq_f32(af, tf);
      vst1q_f32(tPtr, mul);
      oPtr += 4;
      tPtr += 4;
    }

    for (size_t i = 0; i < data_len % 4; ++i)
    {
      tPtr[i] = tPtr[i] * oPtr[i];
    }
  }
}

// Will prioritize row-col addition over col-row addition
void Tensor::add_inplace(const Tensor &other)
{
  if (other.get_dim_len() == 1 && dim_len == 2)
  {
    if (other.get_dims_index(0) == dims[1])
    { // Add row to cols

      float *tPtr = data;
      float *oPtr = other.data;

      // Dont do tiling to match rest of add, can do in future.
      for (size_t i = 0; i < dims[0]; ++i)
      {
        for (size_t j = 0; j + 3 < dims[1]; j += 4)
        {
          float32x4_t tf = vld1q_f32(tPtr);
          float32x4_t af = vld1q_f32(&oPtr[j]);
          float32x4_t add = vaddq_f32(af, tf);
          vst1q_f32(tPtr, add);
          tPtr += 4;
        }
        for (size_t j = 0; j < (dims[1] % 4); ++j)
        {
          (*tPtr) = (*tPtr) + oPtr[j];
          tPtr += 1;
        }
      }
    }
    else if (other.get_dims_index(0) == dims[0])
    { // Add col to rows

      float *tPtr = data;
      float *oPtr = other.data;

      // Dont do tiling to match rest of add, can do in future.
      for (size_t i = 0; i < dims[0]; ++i)
      {
        float vec_float = *oPtr;
        float32x4_t af = vdupq_n_f32(vec_float);
        for (size_t j = 0; j + 3 < dims[1]; j += 4)
        {
          float32x4_t tf = vld1q_f32(tPtr);
          float32x4_t add = vaddq_f32(af, tf);
          vst1q_f32(tPtr, add);
          tPtr += 4;
        }
        for (size_t j = 0; j < (dims[1] % 4); ++j)
        {
          (*tPtr) = (*tPtr) + vec_float;
          tPtr += 1;
        }
        oPtr += 1;
      }
    }
    else
    {
      throw std::invalid_argument("Invalid matrix-vector add dimensions!");
    }
  }
  else
  {
    for (size_t i = 0; i < dim_len; ++i)
    {
      if (dims[i] != other.get_dims_index(i))
      {
        throw std::invalid_argument("Invalid matrix dimensions!");
      }
    }

    const float *oPtr = other.data;
    float *tPtr = data;

    for (size_t i = 0; i + 3 < data_len; i += 4)
    {
      float32x4_t af = vld1q_f32(oPtr);
      float32x4_t tf = vld1q_f32(tPtr);
      float32x4_t add = vaddq_f32(af, tf);
      vst1q_f32(tPtr, add);
      oPtr += 4;
      tPtr += 4;
    }

    for (size_t i = 0; i < data_len % 4; ++i)
    {
      tPtr[i] = tPtr[i] + oPtr[i];
    }
  }
}

void Tensor::apply_inplace(float (*func)(float))
{
  for (size_t i = 0; i < data_len; ++i)
  {
    data[i] = func(data[i]);
  }
}

// Must be 2D and row major. Returns a 1D matrix with sum of all rows.
Tensor Tensor::sum_rows() const
{
  if (dim_len == 2)
  {
    int *new_dims = int_size_alloc(1);
    new_dims[0] = dims[1];

    float *data_out = float_size_alloc(dims[1]);

    float *tPtr = data;
    float *outPtr = data_out;

    // Loop through all cols by 4, using SIMD to add all rows (since rows are
    // contigous)
    for (size_t j = 0; j + 3 < dims[1]; j += 4)
    {
      float32x4_t acc = vdupq_n_f32(0.0f);
      // Use the SIMD accumulator to add all rows for each stride of cols.
      for (size_t i = 0; i < dims[0]; ++i)
      {
        float32x4_t tf = vld1q_f32(tPtr + i * dims[1] + j);
        acc = vaddq_f32(acc, tf);
      }
      vst1q_f32(outPtr, acc);
      outPtr += 4;
    }

    // Scalar cleanup of columns. Generally cache inefficient (high amount of
    // misses), but this is fine for a scalar cleanup as will only happen 3
    // times max.
    for (size_t j = 0; j < (dims[1] % 4); ++j)
    {
      (*outPtr) = 0;
      for (size_t i = 0; i < dims[0]; ++i)
      {
        (*outPtr) += tPtr[i * dims[1] + j];
      }
      outPtr += 1;
    }

    Tensor ret = Tensor(new_dims, 1, data_out, false);
    return ret;
  }
  else
  {
    throw std::invalid_argument("Invalid dimensions, must be 2D");
  }
}

// Must be 2D and row major. Returns a 1D matrix with sum of all cols.
Tensor Tensor::sum_cols() const
{
  if (dim_len == 2)
  {
    int *new_dims = int_size_alloc(1);
    new_dims[0] = dims[0];

    float *data_out = float_size_alloc(dims[0]);

    float *tPtr = data;
    float *outPtr = data_out;

    // Loop through all rows, use SIMD to add since arrays are store in
    // row-major format.
    for (size_t i = 0; i < dims[0]; ++i)
    {
      // SIMD Loop to add all row elements except tail
      float32x4_t acc = vdupq_n_f32(0.0f);
      for (size_t j = 0; j + 3 < dims[1]; j += 4)
      {
        float32x4_t tf = vld1q_f32(tPtr);
        acc = vaddq_f32(acc, tf);
        tPtr += 4;
      }
      (*outPtr) = vaddvq_f32(acc);

      // Scalar tail if needed, will always run at most 3 times.
      for (size_t j = 0; j < (dims[1] % 4); ++j)
      {
        (*outPtr) += *(tPtr);
        tPtr += 1;
      }
      outPtr += 1;
    }

    Tensor ret = Tensor(new_dims, 1, data_out, false);
    return ret;
  }
  else
  {
    throw std::invalid_argument("Invalid dimensions, must be 2D");
  }
}

// General - can improve in future to sum over a given axes, would require
// continous memory for given axes in order to use SIMD
float Tensor::sum() const
{
  float *tPtr = data;
  float out = 0;
  float32x4_t acc = vdupq_n_f32(0.0f);

  // Loop over internal data array and accumulate sum in one SIMD buffer.
  for (size_t i = 0; i + 3 < data_len; i += 4)
  {
    float32x4_t tf = vld1q_f32(tPtr);
    acc = vaddq_f32(acc, tf);
    tPtr += 4;
  }
  // Bring SIMD sum into float
  out = vaddvq_f32(acc);

  for (size_t i = 0; i < data_len % 4; ++i)
  {
    out += tPtr[i];
  }

  return out;
}

float Tensor::get(const std::initializer_list<int> &pos) const
{
  return data[convert_idx(pos)];
}

float Tensor::get_index(int i) const
{
  if (i < 0 || i > data_len - 1)
  {
    throw std::invalid_argument("Invalid index!");
  }
  return data[i];
}

void Tensor::set(const std::initializer_list<int> &pos, float val)
{
  data[convert_idx(pos)] = val;
}

void Tensor::set_index(int i, float val)
{
  if (i < 0 || i > data_len - 1)
  {
    throw std::invalid_argument("Invalid index!");
  }
  data[i] = val;
}

int Tensor::get_dims_index(int i) const
{
  if (i < 0 || i > dim_len - 1)
  {
    throw std::invalid_argument("Invalid index!");
  }
  return dims[i];
}

int Tensor::get_dim_len() const { return dim_len; }

int *Tensor::get_dists_clone() const
{
  int *dists_clone = int_size_alloc(dim_len);
  for (size_t i = 0; i < dim_len; ++i)
  {
    dists_clone[i] = dists[i];
  }
  return dists_clone;
}

int *Tensor::get_dims_clone() const
{
  int *dims_clone = int_size_alloc(dim_len);
  for (size_t i = 0; i < dim_len; ++i)
  {
    dims_clone[i] = dims[i];
  }
  return dims_clone;
}

void Tensor::print_data(int max) const { print_array(data, data_len, max); }

void Tensor::print_dims(int max) const { print_array(dims, dim_len, max); }

void Tensor::print_dists(int max) const { print_array(dists, dim_len, max); }

void Tensor::set_tile(int t) { tile = t; }

int Tensor::get_tile() { return tile; }

int Tensor::get_alignment() { return alignment; }