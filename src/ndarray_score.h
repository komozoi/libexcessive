// Copyright 2021-2026 Komozoi
// Original Creation Date: 2026-8-17
//
// INTERNAL — include only from NDArray.cpp / ndarray_score.cpp.
// Packed score kernels. Implementations live in ndarray_score.cpp.

#ifndef LIBEXCESSIVE_SRC_NDARRAY_SCORE_H
#define LIBEXCESSIVE_SRC_NDARRAY_SCORE_H

#include <cstddef>
#include <cstdint>

enum class NDScoreOp { Dot, L2Self, L2Pair };

// Dense packed pointers (element offset already applied). `b` is unused for L2Self.
float   ndscore_f32(const float* a, const float* b, size_t n, NDScoreOp op);
double  ndscore_f64(const double* a, const double* b, size_t n, NDScoreOp op);
int32_t ndscore_i32(const int32_t* a, const int32_t* b, size_t n, NDScoreOp op);
int64_t ndscore_i64(const int64_t* a, const int64_t* b, size_t n, NDScoreOp op);
int32_t ndscore_u8_i32(const uint8_t* a, const uint8_t* b, size_t n, NDScoreOp op);
int64_t ndscore_u8_i64(const uint8_t* a, const uint8_t* b, size_t n, NDScoreOp op);

// INT3 nibble-packed words. oa/ob are *element* offsets. Widening MAC (3*3 → 9).
int64_t ndscore_int3_i64(const uint64_t* a, size_t oa, const uint64_t* b, size_t ob,
                         size_t n, NDScoreOp op);

// BINARY bit-packed words. oa/ob are *bit* offsets.
// Dot = popcount(AND); L2Self = popcount; L2Pair = Hamming = popcount(XOR).
int64_t ndscore_binary_i64(const uint64_t* a, size_t oa, const uint64_t* b, size_t ob,
                           size_t n, NDScoreOp op);

#endif
