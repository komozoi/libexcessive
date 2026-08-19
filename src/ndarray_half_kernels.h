// INTERNAL — F16 / BF16 convert + arithmetic + score kernels.
// ISA: AVX-512-FP16 / AVX-512-BF16 / AVX-512F cvt / F16C+AVX2 / SSE2 / NEON+fp16.
// Portable path is restrict + 4-wide so GCC/Clang auto-vectorize.

#ifndef LIBEXCESSIVE_SRC_NDARRAY_HALF_KERNELS_H
#define LIBEXCESSIVE_SRC_NDARRAY_HALF_KERNELS_H

#include <cstddef>
#include <cstdint>

enum class NDHalfArith { Add, Sub, Mul, Div };

void ndhalf_f16_to_f32(float* dst, const uint16_t* src, size_t n);
void ndhalf_f32_to_f16(uint16_t* dst, const float* src, size_t n);
void ndhalf_bf16_to_f32(float* dst, const uint16_t* src, size_t n);
void ndhalf_f32_to_bf16(uint16_t* dst, const float* src, size_t n);

void ndhalf_arith_f16(uint16_t* dst, const uint16_t* a, const uint16_t* b,
                      size_t n, NDHalfArith op);
void ndhalf_arith_bf16(uint16_t* dst, const uint16_t* a, const uint16_t* b,
                       size_t n, NDHalfArith op);
void ndhalf_arith_f16_scalar(uint16_t* dst, const uint16_t* a, float s,
                             size_t n, NDHalfArith op);
void ndhalf_arith_bf16_scalar(uint16_t* dst, const uint16_t* a, float s,
                              size_t n, NDHalfArith op);

void ndhalf_min_f16(uint16_t* dst, const uint16_t* a, const uint16_t* b, size_t n);
void ndhalf_max_f16(uint16_t* dst, const uint16_t* a, const uint16_t* b, size_t n);
void ndhalf_min_bf16(uint16_t* dst, const uint16_t* a, const uint16_t* b, size_t n);
void ndhalf_max_bf16(uint16_t* dst, const uint16_t* a, const uint16_t* b, size_t n);

float ndhalf_sum_f16(const uint16_t* a, size_t n);
float ndhalf_sum_bf16(const uint16_t* a, size_t n);
float ndhalf_prod_f16(const uint16_t* a, size_t n);
float ndhalf_prod_bf16(const uint16_t* a, size_t n);
float ndhalf_dot_f16(const uint16_t* a, const uint16_t* b, size_t n);
float ndhalf_dot_bf16(const uint16_t* a, const uint16_t* b, size_t n);
float ndhalf_l2self_f16(const uint16_t* a, size_t n);
float ndhalf_l2self_bf16(const uint16_t* a, size_t n);
float ndhalf_l2pair_f16(const uint16_t* a, const uint16_t* b, size_t n);
float ndhalf_l2pair_bf16(const uint16_t* a, const uint16_t* b, size_t n);

#endif
