
#ifndef WINDOWS_VER
#include "../include/AMXKernel.h"

namespace
{

    // 00000 10 x_mask(5) (0 if all, or num lanes to enable) 00 10 y_mask(5) (0 if all, or num lanes to enable) 00 000 0 z_row(6) 0 x_offset(9, bytes) 0  y_offset(9, bytes)
    uint64_t get_instruction(uint64_t x_mask, uint64_t y_mask, uint64_t z_row, uint64_t x_offset, uint64_t y_offset, bool clear_z)
    {
        return 1ULL << 47 | x_mask << 41 | 1ULL << 38 | y_mask << 32 | (uint64_t)clear_z << 27 | z_row << 20 | x_offset << 10 | y_offset;
    }

    void load_vectors(const float *__restrict__ X, const float *__restrict__ Y)
    {

        constexpr uint64_t instructions = 0b11001ULL << 59;

        uint64_t ldx_operand = instructions | (uint64_t)X;
        AMX_LDX(ldx_operand);

        uint64_t ldy_operand = instructions | (uint64_t)Y;
        AMX_LDY(ldy_operand);
    }
}

void load_amx_output(const float *__restrict__ Z)
{
    for (size_t j = 0; j < 16; j++)
    {
        AMX_LDZ(((uint64_t)(j * 4 + 0) << 56) | (uint64_t)&Z[j * 32]);
        AMX_LDZ(((uint64_t)(j * 4 + 1) << 56) | (uint64_t)&Z[j * 32 + 16]);
        AMX_LDZ(((uint64_t)(j * 4 + 2) << 56) | (uint64_t)&Z[(j + 16) * 32]);
        AMX_LDZ(((uint64_t)(j * 4 + 3) << 56) | (uint64_t)&Z[(j + 16) * 32 + 16]);
    }
}

void store_amx_output(float *__restrict__ Z)
{
    for (size_t j = 0; j < 16; j++)
    {
        AMX_STZ(((uint64_t)(j * 4 + 0) << 56) | (uint64_t)&Z[j * 32]);
        AMX_STZ(((uint64_t)(j * 4 + 1) << 56) | (uint64_t)&Z[j * 32 + 16]);
        AMX_STZ(((uint64_t)(j * 4 + 2) << 56) | (uint64_t)&Z[(j + 16) * 32]);
        AMX_STZ(((uint64_t)(j * 4 + 3) << 56) | (uint64_t)&Z[(j + 16) * 32 + 16]);
    }
}

void amx_kernel_f32(const float *__restrict__ X, const float *__restrict__ Y, size_t mask_x, size_t mask_y, bool clear_z)
{

    load_vectors(X, Y);

    uint64_t bit_x_mask_inner = (mask_x >= 16) ? 0 : mask_x;
    uint64_t bit_x_mask_outer = (mask_x == 32) ? 0 : mask_x - 16;
    uint64_t bit_y_mask_inner = (mask_y >= 16) ? 0 : mask_y;
    uint64_t bit_y_mask_outer = (mask_y == 32) ? 0 : mask_y - 16;

    // FMA at index 1, 1
    if (mask_x > 16 && mask_y > 16)
    {
        uint64_t instr3 = get_instruction(bit_x_mask_outer, bit_y_mask_outer, 3, 64, 64, clear_z);
        AMX_FMA32(instr3);
    }

    // FMA at index 0, 1
    if (mask_y > 16)
    {
        uint64_t instr2 = get_instruction(bit_x_mask_inner, bit_y_mask_outer, 2, 0, 64, clear_z);
        AMX_FMA32(instr2);
    }

    // FMA at index 1, 0
    if (mask_x > 16)
    {
        uint64_t instr1 = get_instruction(bit_x_mask_outer, bit_y_mask_inner, 1, 64, 0, clear_z);
        AMX_FMA32(instr1);
    }

    // FMA at index 0, 0

    uint64_t instr0 = get_instruction(bit_x_mask_inner, bit_y_mask_inner, 0, 0, 0, clear_z);
    AMX_FMA32(instr0);
}
#endif