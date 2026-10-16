#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include "aarch64.h"

void amx_kernel_f32(const float* __restrict__ X, const float* __restrict__ Y, size_t mask_x, size_t mask_y);
void store_amx_output(float* __restrict__ Z);
void load_amx_output(const float*  __restrict__  Z);