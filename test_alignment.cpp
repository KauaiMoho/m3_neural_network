#include <arm_neon.h>
#include <chrono>
#include <stdexcept>
#include <iostream>

constexpr size_t N = 2000000;
constexpr size_t ALIGNMENT = 64;

float sum(const float* data) {
    float32x4_t acc = vdupq_n_f32(0.0f);

    for (size_t i = 0; i + 3 < N; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        acc = vaddq_f32(acc, v);
    }

    return vaddvq_f32(acc);
}

void test(const float* ptr) {

    auto start = std::chrono::high_resolution_clock::now();
    float result = sum(ptr);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> dt = end - start;
    std::cout << dt.count() << "s\n";
}

int main() {

    size_t a = alignof(max_align_t);
    printf("Alignment of max_align_t is %zu (%#zx)\n", a, a); // Malloc only has 8 byte allignment, need to do alligned_alloc

    const size_t bytes = N * sizeof(float) + (N * sizeof(float))%ALIGNMENT;

    float* aligned = static_cast<float*>(tensor_alloc(ALIGNMENT, bytes));

    if (aligned == nullptr) {
        free(aligned);
        return 1;
    }

    for (size_t i = 0; i < N; ++i)
        aligned[i] = 1.1f;

    float* unaligned = static_cast<float*>(malloc(N * sizeof(float)));

    for (size_t i = 0; i < N; ++i)
        unaligned[i] = 1.1f;

    std::cout << "Unaligned: ";
    test(unaligned);

    std::cout << "Aligned: ";
    test(aligned);

    return 0;
}