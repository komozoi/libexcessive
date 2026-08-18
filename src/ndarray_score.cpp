// Copyright 2021-2026 Komozoi
// Original Creation Date: 2026-8-17
//
// Packed score kernels. Prefer auto-vectorizable multi-accumulator loops;
// ISA-specific kernels when the compiler cannot (INT3 nibble MAC, BINARY
// popcount on NEON/AVX-512, F32 FMA when flags are present).

#include "ndarray_score.h"

#include <cstddef>
#include <cstdint>

#if defined(_MSC_VER)
#define NDSCORE_RESTRICT __restrict
#else
#define NDSCORE_RESTRICT __restrict__
#endif

#if defined(__AVX512F__)
#define NDSCORE_AVX512 1
#include <immintrin.h>
#else
#define NDSCORE_AVX512 0
#endif

#if defined(__AVX2__)
#define NDSCORE_AVX2 1
#ifndef _MSC_VER
#include <immintrin.h>
#endif
#else
#define NDSCORE_AVX2 0
#endif

#if defined(__SSE2__)
#define NDSCORE_SSE2 1
#if !NDSCORE_AVX2 && !NDSCORE_AVX512
#include <emmintrin.h>
#endif
#else
#define NDSCORE_SSE2 0
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define NDSCORE_NEON 1
#include <arm_neon.h>
#else
#define NDSCORE_NEON 0
#endif

#if defined(__FMA__) || NDSCORE_AVX512
#define NDSCORE_FMA 1
#else
#define NDSCORE_FMA 0
#endif

#if defined(__AVX512VPOPCNTDQ__)
#define NDSCORE_VPOPCNT 1
#else
#define NDSCORE_VPOPCNT 0
#endif

static size_t popcnt64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
	return (size_t)__builtin_popcountll(x);
#else
	size_t n = 0;
	while (x) {
		n += x & 1u;
		x >>= 1;
	}
	return n;
#endif
}

// ---- auto-vec dense (fallback when no ISA kernel, and for odd Acc widths) --

template <typename Lane, typename Acc>
static Acc dense_auto(const Lane* NDSCORE_RESTRICT a, const Lane* NDSCORE_RESTRICT b,
                      size_t n, NDScoreOp op) {
	Acc s0 = Acc(), s1 = Acc(), s2 = Acc(), s3 = Acc();
	size_t i = 0;
	if (op == NDScoreOp::Dot) {
		for (; i + 4 <= n; i += 4) {
			s0 += Acc(a[i]) * Acc(b[i]);
			s1 += Acc(a[i + 1]) * Acc(b[i + 1]);
			s2 += Acc(a[i + 2]) * Acc(b[i + 2]);
			s3 += Acc(a[i + 3]) * Acc(b[i + 3]);
		}
		Acc s = (s0 + s1) + (s2 + s3);
		for (; i < n; ++i)
			s += Acc(a[i]) * Acc(b[i]);
		return s;
	}
	if (op == NDScoreOp::L2Self) {
		for (; i + 4 <= n; i += 4) {
			Acc v0 = Acc(a[i]), v1 = Acc(a[i + 1]), v2 = Acc(a[i + 2]), v3 = Acc(a[i + 3]);
			s0 += v0 * v0;
			s1 += v1 * v1;
			s2 += v2 * v2;
			s3 += v3 * v3;
		}
		Acc s = (s0 + s1) + (s2 + s3);
		for (; i < n; ++i) {
			Acc v = Acc(a[i]);
			s += v * v;
		}
		return s;
	}
	for (; i + 4 <= n; i += 4) {
		Acc d0 = Acc(a[i]) - Acc(b[i]);
		Acc d1 = Acc(a[i + 1]) - Acc(b[i + 1]);
		Acc d2 = Acc(a[i + 2]) - Acc(b[i + 2]);
		Acc d3 = Acc(a[i + 3]) - Acc(b[i + 3]);
		s0 += d0 * d0;
		s1 += d1 * d1;
		s2 += d2 * d2;
		s3 += d3 * d3;
	}
	Acc s = (s0 + s1) + (s2 + s3);
	for (; i < n; ++i) {
		Acc d = Acc(a[i]) - Acc(b[i]);
		s += d * d;
	}
	return s;
}

// ---- F32 --------------------------------------------------------------------

#if NDSCORE_AVX512
static float f32_avx512(const float* NDSCORE_RESTRICT a, const float* NDSCORE_RESTRICT b,
                        size_t n, NDScoreOp op) {
	__m512 acc0 = _mm512_setzero_ps();
	__m512 acc1 = _mm512_setzero_ps();
	size_t i = 0;
	if (op == NDScoreOp::Dot) {
		for (; i + 32 <= n; i += 32) {
			acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
			acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), acc1);
		}
		for (; i + 16 <= n; i += 16)
			acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
	} else if (op == NDScoreOp::L2Self) {
		for (; i + 32 <= n; i += 32) {
			__m512 x0 = _mm512_loadu_ps(a + i);
			__m512 x1 = _mm512_loadu_ps(a + i + 16);
			acc0 = _mm512_fmadd_ps(x0, x0, acc0);
			acc1 = _mm512_fmadd_ps(x1, x1, acc1);
		}
		for (; i + 16 <= n; i += 16) {
			__m512 x = _mm512_loadu_ps(a + i);
			acc0 = _mm512_fmadd_ps(x, x, acc0);
		}
	} else {
		for (; i + 32 <= n; i += 32) {
			__m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
			__m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16));
			acc0 = _mm512_fmadd_ps(d0, d0, acc0);
			acc1 = _mm512_fmadd_ps(d1, d1, acc1);
		}
		for (; i + 16 <= n; i += 16) {
			__m512 d = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
			acc0 = _mm512_fmadd_ps(d, d, acc0);
		}
	}
	float s = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
	for (; i < n; ++i) {
		if (op == NDScoreOp::Dot)
			s += a[i] * b[i];
		else if (op == NDScoreOp::L2Self)
			s += a[i] * a[i];
		else {
			float d = a[i] - b[i];
			s += d * d;
		}
	}
	return s;
}
#endif

#if NDSCORE_AVX2 && !NDSCORE_AVX512
static inline __m256 f32_madd256(__m256 acc, __m256 x, __m256 y) {
#if NDSCORE_FMA
	return _mm256_fmadd_ps(x, y, acc);
#else
	return _mm256_add_ps(acc, _mm256_mul_ps(x, y));
#endif
}

static float hsum256_ps(__m256 v) {
	__m128 lo = _mm256_castps256_ps128(v);
	__m128 hi = _mm256_extractf128_ps(v, 1);
	__m128 s = _mm_add_ps(lo, hi);
	s = _mm_add_ps(s, _mm_movehl_ps(s, s));
	s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
	return _mm_cvtss_f32(s);
}

static float f32_avx2(const float* NDSCORE_RESTRICT a, const float* NDSCORE_RESTRICT b,
                      size_t n, NDScoreOp op) {
	__m256 acc0 = _mm256_setzero_ps();
	__m256 acc1 = _mm256_setzero_ps();
	__m256 acc2 = _mm256_setzero_ps();
	__m256 acc3 = _mm256_setzero_ps();
	size_t i = 0;
	if (op == NDScoreOp::Dot) {
		for (; i + 32 <= n; i += 32) {
			acc0 = f32_madd256(acc0, _mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
			acc1 = f32_madd256(acc1, _mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
			acc2 = f32_madd256(acc2, _mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
			acc3 = f32_madd256(acc3, _mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
		}
	} else if (op == NDScoreOp::L2Self) {
		for (; i + 32 <= n; i += 32) {
			__m256 x0 = _mm256_loadu_ps(a + i);
			__m256 x1 = _mm256_loadu_ps(a + i + 8);
			__m256 x2 = _mm256_loadu_ps(a + i + 16);
			__m256 x3 = _mm256_loadu_ps(a + i + 24);
			acc0 = f32_madd256(acc0, x0, x0);
			acc1 = f32_madd256(acc1, x1, x1);
			acc2 = f32_madd256(acc2, x2, x2);
			acc3 = f32_madd256(acc3, x3, x3);
		}
	} else {
		for (; i + 32 <= n; i += 32) {
			__m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
			__m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
			__m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
			__m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
			acc0 = f32_madd256(acc0, d0, d0);
			acc1 = f32_madd256(acc1, d1, d1);
			acc2 = f32_madd256(acc2, d2, d2);
			acc3 = f32_madd256(acc3, d3, d3);
		}
	}
	float s = hsum256_ps(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
	for (; i < n; ++i) {
		if (op == NDScoreOp::Dot)
			s += a[i] * b[i];
		else if (op == NDScoreOp::L2Self)
			s += a[i] * a[i];
		else {
			float d = a[i] - b[i];
			s += d * d;
		}
	}
	return s;
}
#endif

#if NDSCORE_SSE2 && !NDSCORE_AVX2 && !NDSCORE_AVX512
static inline __m128 f32_madd128(__m128 acc, __m128 x, __m128 y) {
#if NDSCORE_FMA
	return _mm_fmadd_ps(x, y, acc);
#else
	return _mm_add_ps(acc, _mm_mul_ps(x, y));
#endif
}

static float hsum128_ps(__m128 v) {
	__m128 s = _mm_add_ps(v, _mm_movehl_ps(v, v));
	s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
	return _mm_cvtss_f32(s);
}

static float f32_sse(const float* NDSCORE_RESTRICT a, const float* NDSCORE_RESTRICT b,
                     size_t n, NDScoreOp op) {
	__m128 acc0 = _mm_setzero_ps();
	__m128 acc1 = _mm_setzero_ps();
	__m128 acc2 = _mm_setzero_ps();
	__m128 acc3 = _mm_setzero_ps();
	size_t i = 0;
	if (op == NDScoreOp::Dot) {
		for (; i + 16 <= n; i += 16) {
			acc0 = f32_madd128(acc0, _mm_loadu_ps(a + i), _mm_loadu_ps(b + i));
			acc1 = f32_madd128(acc1, _mm_loadu_ps(a + i + 4), _mm_loadu_ps(b + i + 4));
			acc2 = f32_madd128(acc2, _mm_loadu_ps(a + i + 8), _mm_loadu_ps(b + i + 8));
			acc3 = f32_madd128(acc3, _mm_loadu_ps(a + i + 12), _mm_loadu_ps(b + i + 12));
		}
	} else if (op == NDScoreOp::L2Self) {
		for (; i + 16 <= n; i += 16) {
			__m128 x0 = _mm_loadu_ps(a + i);
			__m128 x1 = _mm_loadu_ps(a + i + 4);
			__m128 x2 = _mm_loadu_ps(a + i + 8);
			__m128 x3 = _mm_loadu_ps(a + i + 12);
			acc0 = f32_madd128(acc0, x0, x0);
			acc1 = f32_madd128(acc1, x1, x1);
			acc2 = f32_madd128(acc2, x2, x2);
			acc3 = f32_madd128(acc3, x3, x3);
		}
	} else {
		for (; i + 16 <= n; i += 16) {
			__m128 d0 = _mm_sub_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i));
			__m128 d1 = _mm_sub_ps(_mm_loadu_ps(a + i + 4), _mm_loadu_ps(b + i + 4));
			__m128 d2 = _mm_sub_ps(_mm_loadu_ps(a + i + 8), _mm_loadu_ps(b + i + 8));
			__m128 d3 = _mm_sub_ps(_mm_loadu_ps(a + i + 12), _mm_loadu_ps(b + i + 12));
			acc0 = f32_madd128(acc0, d0, d0);
			acc1 = f32_madd128(acc1, d1, d1);
			acc2 = f32_madd128(acc2, d2, d2);
			acc3 = f32_madd128(acc3, d3, d3);
		}
	}
	float s = hsum128_ps(_mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3)));
	for (; i < n; ++i) {
		if (op == NDScoreOp::Dot)
			s += a[i] * b[i];
		else if (op == NDScoreOp::L2Self)
			s += a[i] * a[i];
		else {
			float d = a[i] - b[i];
			s += d * d;
		}
	}
	return s;
}
#endif

#if NDSCORE_NEON
#if defined(__ARM_FEATURE_FMA) || defined(__aarch64__)
#define ndscore_vfmaq_f32(acc, x, y) vfmaq_f32((acc), (x), (y))
#else
#define ndscore_vfmaq_f32(acc, x, y) vmlaq_f32((acc), (x), (y))
#endif

static float hsum_f32x4(float32x4_t v) {
#if defined(__aarch64__)
	return vaddvq_f32(v);
#else
	float32x2_t p = vadd_f32(vget_low_f32(v), vget_high_f32(v));
	p = vpadd_f32(p, p);
	return vget_lane_f32(p, 0);
#endif
}

static float f32_neon(const float* NDSCORE_RESTRICT a, const float* NDSCORE_RESTRICT b,
                      size_t n, NDScoreOp op) {
	float32x4_t acc0 = vdupq_n_f32(0.0f);
	float32x4_t acc1 = vdupq_n_f32(0.0f);
	float32x4_t acc2 = vdupq_n_f32(0.0f);
	float32x4_t acc3 = vdupq_n_f32(0.0f);
	size_t i = 0;
	if (op == NDScoreOp::Dot) {
		for (; i + 16 <= n; i += 16) {
			acc0 = ndscore_vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
			acc1 = ndscore_vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
			acc2 = ndscore_vfmaq_f32(acc2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
			acc3 = ndscore_vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
		}
	} else if (op == NDScoreOp::L2Self) {
		for (; i + 16 <= n; i += 16) {
			float32x4_t x0 = vld1q_f32(a + i);
			float32x4_t x1 = vld1q_f32(a + i + 4);
			float32x4_t x2 = vld1q_f32(a + i + 8);
			float32x4_t x3 = vld1q_f32(a + i + 12);
			acc0 = ndscore_vfmaq_f32(acc0, x0, x0);
			acc1 = ndscore_vfmaq_f32(acc1, x1, x1);
			acc2 = ndscore_vfmaq_f32(acc2, x2, x2);
			acc3 = ndscore_vfmaq_f32(acc3, x3, x3);
		}
	} else {
		for (; i + 16 <= n; i += 16) {
			float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
			float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
			float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
			float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
			acc0 = ndscore_vfmaq_f32(acc0, d0, d0);
			acc1 = ndscore_vfmaq_f32(acc1, d1, d1);
			acc2 = ndscore_vfmaq_f32(acc2, d2, d2);
			acc3 = ndscore_vfmaq_f32(acc3, d3, d3);
		}
	}
	float s = hsum_f32x4(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
	for (; i < n; ++i) {
		if (op == NDScoreOp::Dot)
			s += a[i] * b[i];
		else if (op == NDScoreOp::L2Self)
			s += a[i] * a[i];
		else {
			float d = a[i] - b[i];
			s += d * d;
		}
	}
	return s;
}
#endif

float ndscore_f32(const float* a, const float* b, size_t n, NDScoreOp op) {
#if NDSCORE_AVX512
	return f32_avx512(a, b, n, op);
#elif NDSCORE_AVX2
	return f32_avx2(a, b, n, op);
#elif NDSCORE_NEON
	return f32_neon(a, b, n, op);
#elif NDSCORE_SSE2
	return f32_sse(a, b, n, op);
#else
	return dense_auto<float, float>(a, b, n, op);
#endif
}

// ---- F64 --------------------------------------------------------------------

#if NDSCORE_AVX512
static double f64_avx512(const double* NDSCORE_RESTRICT a, const double* NDSCORE_RESTRICT b,
                         size_t n, NDScoreOp op) {
	__m512d acc0 = _mm512_setzero_pd();
	__m512d acc1 = _mm512_setzero_pd();
	size_t i = 0;
	if (op == NDScoreOp::Dot) {
		for (; i + 16 <= n; i += 16) {
			acc0 = _mm512_fmadd_pd(_mm512_loadu_pd(a + i), _mm512_loadu_pd(b + i), acc0);
			acc1 = _mm512_fmadd_pd(_mm512_loadu_pd(a + i + 8), _mm512_loadu_pd(b + i + 8), acc1);
		}
	} else if (op == NDScoreOp::L2Self) {
		for (; i + 16 <= n; i += 16) {
			__m512d x0 = _mm512_loadu_pd(a + i);
			__m512d x1 = _mm512_loadu_pd(a + i + 8);
			acc0 = _mm512_fmadd_pd(x0, x0, acc0);
			acc1 = _mm512_fmadd_pd(x1, x1, acc1);
		}
	} else {
		for (; i + 16 <= n; i += 16) {
			__m512d d0 = _mm512_sub_pd(_mm512_loadu_pd(a + i), _mm512_loadu_pd(b + i));
			__m512d d1 = _mm512_sub_pd(_mm512_loadu_pd(a + i + 8), _mm512_loadu_pd(b + i + 8));
			acc0 = _mm512_fmadd_pd(d0, d0, acc0);
			acc1 = _mm512_fmadd_pd(d1, d1, acc1);
		}
	}
	double s = _mm512_reduce_add_pd(_mm512_add_pd(acc0, acc1));
	for (; i < n; ++i) {
		if (op == NDScoreOp::Dot)
			s += a[i] * b[i];
		else if (op == NDScoreOp::L2Self)
			s += a[i] * a[i];
		else {
			double d = a[i] - b[i];
			s += d * d;
		}
	}
	return s;
}
#endif

#if NDSCORE_AVX2 && !NDSCORE_AVX512
static inline __m256d f64_madd256(__m256d acc, __m256d x, __m256d y) {
#if NDSCORE_FMA
	return _mm256_fmadd_pd(x, y, acc);
#else
	return _mm256_add_pd(acc, _mm256_mul_pd(x, y));
#endif
}

static double hsum256_pd(__m256d v) {
	__m128d lo = _mm256_castpd256_pd128(v);
	__m128d hi = _mm256_extractf128_pd(v, 1);
	__m128d s = _mm_add_pd(lo, hi);
	s = _mm_add_sd(s, _mm_unpackhi_pd(s, s));
	return _mm_cvtsd_f64(s);
}

static double f64_avx2(const double* NDSCORE_RESTRICT a, const double* NDSCORE_RESTRICT b,
                       size_t n, NDScoreOp op) {
	__m256d acc0 = _mm256_setzero_pd();
	__m256d acc1 = _mm256_setzero_pd();
	size_t i = 0;
	if (op == NDScoreOp::Dot) {
		for (; i + 8 <= n; i += 8) {
			acc0 = f64_madd256(acc0, _mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i));
			acc1 = f64_madd256(acc1, _mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(b + i + 4));
		}
	} else if (op == NDScoreOp::L2Self) {
		for (; i + 8 <= n; i += 8) {
			__m256d x0 = _mm256_loadu_pd(a + i);
			__m256d x1 = _mm256_loadu_pd(a + i + 4);
			acc0 = f64_madd256(acc0, x0, x0);
			acc1 = f64_madd256(acc1, x1, x1);
		}
	} else {
		for (; i + 8 <= n; i += 8) {
			__m256d d0 = _mm256_sub_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i));
			__m256d d1 = _mm256_sub_pd(_mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(b + i + 4));
			acc0 = f64_madd256(acc0, d0, d0);
			acc1 = f64_madd256(acc1, d1, d1);
		}
	}
	double s = hsum256_pd(_mm256_add_pd(acc0, acc1));
	for (; i < n; ++i) {
		if (op == NDScoreOp::Dot)
			s += a[i] * b[i];
		else if (op == NDScoreOp::L2Self)
			s += a[i] * a[i];
		else {
			double d = a[i] - b[i];
			s += d * d;
		}
	}
	return s;
}
#endif

#if NDSCORE_NEON && defined(__aarch64__)
static double f64_neon(const double* NDSCORE_RESTRICT a, const double* NDSCORE_RESTRICT b,
                       size_t n, NDScoreOp op) {
	float64x2_t acc0 = vdupq_n_f64(0.0);
	float64x2_t acc1 = vdupq_n_f64(0.0);
	size_t i = 0;
	if (op == NDScoreOp::Dot) {
		for (; i + 4 <= n; i += 4) {
			acc0 = vfmaq_f64(acc0, vld1q_f64(a + i), vld1q_f64(b + i));
			acc1 = vfmaq_f64(acc1, vld1q_f64(a + i + 2), vld1q_f64(b + i + 2));
		}
	} else if (op == NDScoreOp::L2Self) {
		for (; i + 4 <= n; i += 4) {
			float64x2_t x0 = vld1q_f64(a + i);
			float64x2_t x1 = vld1q_f64(a + i + 2);
			acc0 = vfmaq_f64(acc0, x0, x0);
			acc1 = vfmaq_f64(acc1, x1, x1);
		}
	} else {
		for (; i + 4 <= n; i += 4) {
			float64x2_t d0 = vsubq_f64(vld1q_f64(a + i), vld1q_f64(b + i));
			float64x2_t d1 = vsubq_f64(vld1q_f64(a + i + 2), vld1q_f64(b + i + 2));
			acc0 = vfmaq_f64(acc0, d0, d0);
			acc1 = vfmaq_f64(acc1, d1, d1);
		}
	}
	double s = vaddvq_f64(vaddq_f64(acc0, acc1));
	for (; i < n; ++i) {
		if (op == NDScoreOp::Dot)
			s += a[i] * b[i];
		else if (op == NDScoreOp::L2Self)
			s += a[i] * a[i];
		else {
			double d = a[i] - b[i];
			s += d * d;
		}
	}
	return s;
}
#endif

double ndscore_f64(const double* a, const double* b, size_t n, NDScoreOp op) {
#if NDSCORE_AVX512
	return f64_avx512(a, b, n, op);
#elif NDSCORE_AVX2
	return f64_avx2(a, b, n, op);
#elif NDSCORE_NEON && defined(__aarch64__)
	return f64_neon(a, b, n, op);
#else
	return dense_auto<double, double>(a, b, n, op);
#endif
}

// ---- INT32 / INT64 / UINT8 (ISA where it beats auto-vec; else 4-acc) --------

#if NDSCORE_AVX512
static int32_t i32_avx512(const int32_t* NDSCORE_RESTRICT a, const int32_t* NDSCORE_RESTRICT b,
                          size_t n, NDScoreOp op) {
	__m512i acc0 = _mm512_setzero_si512();
	__m512i acc1 = _mm512_setzero_si512();
	size_t i = 0;
	if (op == NDScoreOp::Dot) {
		for (; i + 32 <= n; i += 32) {
			acc0 = _mm512_add_epi32(acc0, _mm512_mullo_epi32(_mm512_loadu_si512(a + i),
			                                                 _mm512_loadu_si512(b + i)));
			acc1 = _mm512_add_epi32(acc1, _mm512_mullo_epi32(_mm512_loadu_si512(a + i + 16),
			                                                 _mm512_loadu_si512(b + i + 16)));
		}
	} else if (op == NDScoreOp::L2Self) {
		for (; i + 32 <= n; i += 32) {
			__m512i x0 = _mm512_loadu_si512(a + i);
			__m512i x1 = _mm512_loadu_si512(a + i + 16);
			acc0 = _mm512_add_epi32(acc0, _mm512_mullo_epi32(x0, x0));
			acc1 = _mm512_add_epi32(acc1, _mm512_mullo_epi32(x1, x1));
		}
	} else {
		for (; i + 32 <= n; i += 32) {
			__m512i d0 = _mm512_sub_epi32(_mm512_loadu_si512(a + i), _mm512_loadu_si512(b + i));
			__m512i d1 = _mm512_sub_epi32(_mm512_loadu_si512(a + i + 16), _mm512_loadu_si512(b + i + 16));
			acc0 = _mm512_add_epi32(acc0, _mm512_mullo_epi32(d0, d0));
			acc1 = _mm512_add_epi32(acc1, _mm512_mullo_epi32(d1, d1));
		}
	}
	int32_t s = _mm512_reduce_add_epi32(_mm512_add_epi32(acc0, acc1));
	for (; i < n; ++i) {
		if (op == NDScoreOp::Dot)
			s += a[i] * b[i];
		else if (op == NDScoreOp::L2Self)
			s += a[i] * a[i];
		else {
			int32_t d = a[i] - b[i];
			s += d * d;
		}
	}
	return s;
}

static int32_t u8_avx512(const uint8_t* NDSCORE_RESTRICT a, const uint8_t* NDSCORE_RESTRICT b,
                         size_t n, NDScoreOp op) {
	__m512i acc = _mm512_setzero_si512();
	size_t i = 0;
	for (; i + 32 <= n; i += 32) {
		__m256i av = _mm256_loadu_si256((const __m256i*)(a + i));
		__m256i bv = op == NDScoreOp::L2Self ? av : _mm256_loadu_si256((const __m256i*)(b + i));
		__m512i aw = _mm512_cvtepu8_epi16(av);
		__m512i bw = _mm512_cvtepu8_epi16(bv);
		if (op == NDScoreOp::L2Pair)
			bw = _mm512_sub_epi16(aw, bw), aw = bw;
		acc = _mm512_add_epi32(acc, _mm512_madd_epi16(aw, op == NDScoreOp::L2Pair ? aw : bw));
	}
	int32_t s = _mm512_reduce_add_epi32(acc);
	for (; i < n; ++i) {
		int32_t va = (int32_t)a[i];
		int32_t vb = (op == NDScoreOp::L2Self) ? va : (int32_t)b[i];
		if (op == NDScoreOp::L2Pair) {
			int32_t d = va - vb;
			s += d * d;
		} else
			s += va * vb;
	}
	return s;
}
#endif

#if NDSCORE_AVX2 && !NDSCORE_AVX512
static int32_t hsum256_epi32(__m256i v) {
	__m128i lo = _mm256_castsi256_si128(v);
	__m128i hi = _mm256_extracti128_si256(v, 1);
	__m128i s = _mm_add_epi32(lo, hi);
	s = _mm_hadd_epi32(s, s);
	s = _mm_hadd_epi32(s, s);
	return _mm_cvtsi128_si32(s);
}

static int32_t u8_avx2(const uint8_t* NDSCORE_RESTRICT a, const uint8_t* NDSCORE_RESTRICT b,
                       size_t n, NDScoreOp op) {
	__m256i acc = _mm256_setzero_si256();
	size_t i = 0;
	for (; i + 16 <= n; i += 16) {
		__m128i av = _mm_loadu_si128((const __m128i*)(a + i));
		__m128i bv = op == NDScoreOp::L2Self ? av : _mm_loadu_si128((const __m128i*)(b + i));
		__m256i aw = _mm256_cvtepu8_epi16(av);
		__m256i bw = _mm256_cvtepu8_epi16(bv);
		if (op == NDScoreOp::L2Pair)
			bw = _mm256_sub_epi16(aw, bw), aw = bw;
		acc = _mm256_add_epi32(acc, _mm256_madd_epi16(aw, op == NDScoreOp::L2Pair ? aw : bw));
	}
	int32_t s = hsum256_epi32(acc);
	for (; i < n; ++i) {
		int32_t va = (int32_t)a[i];
		int32_t vb = (op == NDScoreOp::L2Self) ? va : (int32_t)b[i];
		if (op == NDScoreOp::L2Pair) {
			int32_t d = va - vb;
			s += d * d;
		} else
			s += va * vb;
	}
	return s;
}
#endif

#if NDSCORE_NEON
static int32_t u8_neon(const uint8_t* NDSCORE_RESTRICT a, const uint8_t* NDSCORE_RESTRICT b,
                       size_t n, NDScoreOp op) {
	int32x4_t acc0 = vdupq_n_s32(0);
	int32x4_t acc1 = vdupq_n_s32(0);
	size_t i = 0;
	for (; i + 16 <= n; i += 16) {
		uint8x16_t av = vld1q_u8(a + i);
		uint8x16_t bv = (op == NDScoreOp::L2Self) ? av : vld1q_u8(b + i);
		int16x8_t a_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(av)));
		int16x8_t a_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(av)));
		int16x8_t b_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bv)));
		int16x8_t b_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(bv)));
		if (op == NDScoreOp::L2Pair) {
			b_lo = vsubq_s16(a_lo, b_lo);
			b_hi = vsubq_s16(a_hi, b_hi);
			a_lo = b_lo;
			a_hi = b_hi;
		}
		acc0 = vmlal_s16(acc0, vget_low_s16(a_lo), vget_low_s16(op == NDScoreOp::L2Pair ? a_lo : b_lo));
		acc1 = vmlal_s16(acc1, vget_high_s16(a_lo), vget_high_s16(op == NDScoreOp::L2Pair ? a_lo : b_lo));
		acc0 = vmlal_s16(acc0, vget_low_s16(a_hi), vget_low_s16(op == NDScoreOp::L2Pair ? a_hi : b_hi));
		acc1 = vmlal_s16(acc1, vget_high_s16(a_hi), vget_high_s16(op == NDScoreOp::L2Pair ? a_hi : b_hi));
	}
#if defined(__aarch64__)
	int32_t s = vaddvq_s32(vaddq_s32(acc0, acc1));
#else
	int32x2_t p = vadd_s32(vget_low_s32(vaddq_s32(acc0, acc1)), vget_high_s32(vaddq_s32(acc0, acc1)));
	int32_t s = vget_lane_s32(vpadd_s32(p, p), 0);
#endif
	for (; i < n; ++i) {
		int32_t va = (int32_t)a[i];
		int32_t vb = (op == NDScoreOp::L2Self) ? va : (int32_t)b[i];
		if (op == NDScoreOp::L2Pair) {
			int32_t d = va - vb;
			s += d * d;
		} else
			s += va * vb;
	}
	return s;
}
#endif

int32_t ndscore_i32(const int32_t* a, const int32_t* b, size_t n, NDScoreOp op) {
#if NDSCORE_AVX512
	return i32_avx512(a, b, n, op);
#else
	return dense_auto<int32_t, int32_t>(a, b, n, op);
#endif
}

int64_t ndscore_i64(const int64_t* a, const int64_t* b, size_t n, NDScoreOp op) {
	return dense_auto<int64_t, int64_t>(a, b, n, op);
}

int32_t ndscore_u8_i32(const uint8_t* a, const uint8_t* b, size_t n, NDScoreOp op) {
#if NDSCORE_AVX512
	return u8_avx512(a, b, n, op);
#elif NDSCORE_AVX2
	return u8_avx2(a, b, n, op);
#elif NDSCORE_NEON
	return u8_neon(a, b, n, op);
#else
	return dense_auto<uint8_t, int32_t>(a, b, n, op);
#endif
}

int64_t ndscore_u8_i64(const uint8_t* a, const uint8_t* b, size_t n, NDScoreOp op) {
	return (int64_t)ndscore_u8_i32(a, b, n, op);
}

// ---- INT3 widening MAC ------------------------------------------------------

static int int3_lane(const uint64_t* words, size_t i) {
	uint8_t p = (uint8_t)((words[i / 16] >> ((i % 16) * 4)) & 7u);
	return (p >= 4) ? (int)p - 8 : (int)p;
}

static uint64_t int3_tail_mask(size_t remLanes) {
	if (remLanes >= 16)
		return ~uint64_t(0);
	return (uint64_t(1) << (remLanes * 4)) - 1u;
}

static int64_t int3_scalar(const uint64_t* a, size_t oa, const uint64_t* b, size_t ob,
                           size_t n, NDScoreOp op) {
	int64_t acc = 0;
	for (size_t i = 0; i < n; ++i) {
		int va = int3_lane(a, oa + i);
		if (op == NDScoreOp::L2Self) {
			acc += (int64_t)va * va;
			continue;
		}
		int vb = int3_lane(b, ob + i);
		if (op == NDScoreOp::Dot)
			acc += (int64_t)va * vb;
		else {
			int d = va - vb;
			acc += (int64_t)d * d;
		}
	}
	return acc;
}

#if NDSCORE_SSE2 || NDSCORE_AVX2 || NDSCORE_AVX512
// 16 signed 3-bit lanes from one packed word → i8 in xmm.
static inline __m128i int3_unpack_word(uint64_t word) {
	__m128i evens = _mm_cvtsi64_si128((long long)(word & 0x0f0f0f0f0f0f0f0fULL));
	__m128i odds = _mm_cvtsi64_si128((long long)((word >> 4) & 0x0f0f0f0f0f0f0f0fULL));
	__m128i bytes = _mm_unpacklo_epi8(evens, odds);
	__m128i three = _mm_and_si128(bytes, _mm_set1_epi8(7));
	__m128i neg = _mm_cmpgt_epi8(three, _mm_set1_epi8(3));
	return _mm_add_epi8(three, _mm_and_si128(neg, _mm_set1_epi8((char)-8)));
}

#if NDSCORE_AVX2 || NDSCORE_AVX512
static int64_t int3_avx2_words(const uint64_t* a, const uint64_t* b, size_t nWords, size_t rem,
                               NDScoreOp op) {
	__m256i acc = _mm256_setzero_si256();
	size_t w = 0;
	const size_t full = rem ? nWords - 1 : nWords;
	for (; w < full; ++w) {
		__m256i left16 = _mm256_cvtepi8_epi16(int3_unpack_word(a[w]));
		if (op == NDScoreOp::L2Self) {
			acc = _mm256_add_epi32(acc, _mm256_madd_epi16(left16, left16));
			continue;
		}
		__m256i right16 = _mm256_cvtepi8_epi16(int3_unpack_word(b[w]));
		if (op == NDScoreOp::Dot)
			acc = _mm256_add_epi32(acc, _mm256_madd_epi16(left16, right16));
		else {
			__m256i d = _mm256_sub_epi16(left16, right16);
			acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d, d));
		}
	}
	if (rem) {
		uint64_t mask = int3_tail_mask(rem);
		__m256i left16 = _mm256_cvtepi8_epi16(int3_unpack_word(a[w] & mask));
		if (op == NDScoreOp::L2Self)
			acc = _mm256_add_epi32(acc, _mm256_madd_epi16(left16, left16));
		else {
			__m256i right16 = _mm256_cvtepi8_epi16(int3_unpack_word(b[w] & mask));
			if (op == NDScoreOp::Dot)
				acc = _mm256_add_epi32(acc, _mm256_madd_epi16(left16, right16));
			else {
				__m256i d = _mm256_sub_epi16(left16, right16);
				acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d, d));
			}
		}
	}
	__m128i lo = _mm256_castsi256_si128(acc);
	__m128i hi = _mm256_extracti128_si256(acc, 1);
	__m128i s = _mm_add_epi32(lo, hi);
	s = _mm_hadd_epi32(s, s);
	s = _mm_hadd_epi32(s, s);
	return (int64_t)_mm_cvtsi128_si32(s);
}
#endif

#if NDSCORE_SSE2 && !NDSCORE_AVX2 && !NDSCORE_AVX512
static inline __m128i int3_widen_lo(__m128i bytes) {
	__m128i sign = _mm_cmpgt_epi8(_mm_setzero_si128(), bytes);
	return _mm_unpacklo_epi8(bytes, sign);
}
static inline __m128i int3_widen_hi(__m128i bytes) {
	__m128i sign = _mm_cmpgt_epi8(_mm_setzero_si128(), bytes);
	return _mm_unpackhi_epi8(bytes, sign);
}

static int64_t int3_sse_words(const uint64_t* a, const uint64_t* b, size_t nWords, size_t rem,
                              NDScoreOp op) {
	__m128i acc = _mm_setzero_si128();
	size_t w = 0;
	const size_t full = rem ? nWords - 1 : nWords;
	for (; w < full; ++w) {
		__m128i left = int3_unpack_word(a[w]);
		__m128i l0 = int3_widen_lo(left), l1 = int3_widen_hi(left);
		if (op == NDScoreOp::L2Self) {
			acc = _mm_add_epi32(acc, _mm_madd_epi16(l0, l0));
			acc = _mm_add_epi32(acc, _mm_madd_epi16(l1, l1));
			continue;
		}
		__m128i right = int3_unpack_word(b[w]);
		__m128i r0 = int3_widen_lo(right), r1 = int3_widen_hi(right);
		if (op == NDScoreOp::Dot) {
			acc = _mm_add_epi32(acc, _mm_madd_epi16(l0, r0));
			acc = _mm_add_epi32(acc, _mm_madd_epi16(l1, r1));
		} else {
			__m128i d0 = _mm_sub_epi16(l0, r0), d1 = _mm_sub_epi16(l1, r1);
			acc = _mm_add_epi32(acc, _mm_madd_epi16(d0, d0));
			acc = _mm_add_epi32(acc, _mm_madd_epi16(d1, d1));
		}
	}
	if (rem) {
		uint64_t mask = int3_tail_mask(rem);
		__m128i left = int3_unpack_word(a[w] & mask);
		__m128i l0 = int3_widen_lo(left), l1 = int3_widen_hi(left);
		if (op == NDScoreOp::L2Self) {
			acc = _mm_add_epi32(acc, _mm_madd_epi16(l0, l0));
			acc = _mm_add_epi32(acc, _mm_madd_epi16(l1, l1));
		} else {
			__m128i right = int3_unpack_word(b[w] & mask);
			__m128i r0 = int3_widen_lo(right), r1 = int3_widen_hi(right);
			if (op == NDScoreOp::Dot) {
				acc = _mm_add_epi32(acc, _mm_madd_epi16(l0, r0));
				acc = _mm_add_epi32(acc, _mm_madd_epi16(l1, r1));
			} else {
				__m128i d0 = _mm_sub_epi16(l0, r0), d1 = _mm_sub_epi16(l1, r1);
				acc = _mm_add_epi32(acc, _mm_madd_epi16(d0, d0));
				acc = _mm_add_epi32(acc, _mm_madd_epi16(d1, d1));
			}
		}
	}
	acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 8));
	acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 4));
	return (int64_t)_mm_cvtsi128_si32(acc);
}
#endif
#endif // x86 INT3

#if NDSCORE_NEON
static int8x16_t int3_unpack_word_neon(uint64_t word) {
	uint8x8_t evens = vcreate_u8(word & 0x0f0f0f0f0f0f0f0fULL);
	uint8x8_t odds = vcreate_u8((word >> 4) & 0x0f0f0f0f0f0f0f0fULL);
	uint8x8x2_t z = vzip_u8(evens, odds);
	uint8x16_t bytes = vcombine_u8(z.val[0], z.val[1]);
	int8x16_t three = vreinterpretq_s8_u8(vandq_u8(bytes, vdupq_n_u8(7)));
	uint8x16_t neg = vcgtq_s8(three, vdupq_n_s8(3));
	return vaddq_s8(three, vandq_s8(vreinterpretq_s8_u8(neg), vdupq_n_s8(-8)));
}

static int64_t int3_neon_words(const uint64_t* a, const uint64_t* b, size_t nWords, size_t rem,
                               NDScoreOp op) {
	int32x4_t acc0 = vdupq_n_s32(0);
	int32x4_t acc1 = vdupq_n_s32(0);
	size_t w = 0;
	const size_t full = rem ? nWords - 1 : nWords;
	for (; w < full; ++w) {
		int8x16_t left = int3_unpack_word_neon(a[w]);
		int16x8_t l0 = vmovl_s8(vget_low_s8(left));
		int16x8_t l1 = vmovl_s8(vget_high_s8(left));
		int16x8_t r0, r1;
		if (op == NDScoreOp::L2Self) {
			r0 = l0;
			r1 = l1;
		} else {
			int8x16_t right = int3_unpack_word_neon(b[w]);
			r0 = vmovl_s8(vget_low_s8(right));
			r1 = vmovl_s8(vget_high_s8(right));
			if (op == NDScoreOp::L2Pair) {
				r0 = vsubq_s16(l0, r0);
				r1 = vsubq_s16(l1, r1);
				l0 = r0;
				l1 = r1;
			}
		}
		acc0 = vmlal_s16(acc0, vget_low_s16(l0), vget_low_s16(r0));
		acc1 = vmlal_s16(acc1, vget_high_s16(l0), vget_high_s16(r0));
		acc0 = vmlal_s16(acc0, vget_low_s16(l1), vget_low_s16(r1));
		acc1 = vmlal_s16(acc1, vget_high_s16(l1), vget_high_s16(r1));
	}
	if (rem) {
		uint64_t mask = int3_tail_mask(rem);
		int8x16_t left = int3_unpack_word_neon(a[w] & mask);
		int16x8_t l0 = vmovl_s8(vget_low_s8(left));
		int16x8_t l1 = vmovl_s8(vget_high_s8(left));
		int16x8_t r0, r1;
		if (op == NDScoreOp::L2Self) {
			r0 = l0;
			r1 = l1;
		} else {
			int8x16_t right = int3_unpack_word_neon(b[w] & mask);
			r0 = vmovl_s8(vget_low_s8(right));
			r1 = vmovl_s8(vget_high_s8(right));
			if (op == NDScoreOp::L2Pair) {
				r0 = vsubq_s16(l0, r0);
				r1 = vsubq_s16(l1, r1);
				l0 = r0;
				l1 = r1;
			}
		}
		acc0 = vmlal_s16(acc0, vget_low_s16(l0), vget_low_s16(r0));
		acc1 = vmlal_s16(acc1, vget_high_s16(l0), vget_high_s16(r0));
		acc0 = vmlal_s16(acc0, vget_low_s16(l1), vget_low_s16(r1));
		acc1 = vmlal_s16(acc1, vget_high_s16(l1), vget_high_s16(r1));
	}
#if defined(__aarch64__)
	return (int64_t)vaddvq_s32(vaddq_s32(acc0, acc1));
#else
	int32x2_t p = vadd_s32(vget_low_s32(vaddq_s32(acc0, acc1)), vget_high_s32(vaddq_s32(acc0, acc1)));
	return (int64_t)vget_lane_s32(vpadd_s32(p, p), 0);
#endif
}
#endif

int64_t ndscore_int3_i64(const uint64_t* a, size_t oa, const uint64_t* b, size_t ob,
                         size_t n, NDScoreOp op) {
	if (n == 0)
		return 0;
	if ((oa % 16) != 0 || (ob % 16) != 0)
		return int3_scalar(a, oa, b, ob, n, op);
	const uint64_t* aw = a + oa / 16;
	const uint64_t* bw = (op == NDScoreOp::L2Self) ? aw : (b + ob / 16);
	const size_t nWords = (n + 15) / 16;
	const size_t rem = n % 16;
#if NDSCORE_AVX2 || NDSCORE_AVX512
	return int3_avx2_words(aw, bw, nWords, rem, op);
#elif NDSCORE_NEON
	return int3_neon_words(aw, bw, nWords, rem, op);
#elif NDSCORE_SSE2
	return int3_sse_words(aw, bw, nWords, rem, op);
#else
	return int3_scalar(a, oa, b, ob, n, op);
#endif
}

// ---- BINARY popcount --------------------------------------------------------

static int64_t binary_scalar_bits(const uint64_t* a, size_t oa, const uint64_t* b, size_t ob,
                                  size_t n, NDScoreOp op) {
	int64_t acc = 0;
	for (size_t i = 0; i < n; ++i) {
		uint8_t ba = (uint8_t)((a[(oa + i) >> 6] >> ((oa + i) & 63)) & 1u);
		if (op == NDScoreOp::L2Self) {
			acc += ba;
			continue;
		}
		uint8_t bb = (uint8_t)((b[(ob + i) >> 6] >> ((ob + i) & 63)) & 1u);
		acc += (op == NDScoreOp::Dot) ? (ba & bb) : (ba ^ bb);
	}
	return acc;
}

static int64_t binary_words_pop(const uint64_t* NDSCORE_RESTRICT a, const uint64_t* NDSCORE_RESTRICT b,
                                size_t nWords, size_t remBits, NDScoreOp op) {
	int64_t acc = 0;
	size_t w = 0;
	const size_t fullWords = remBits ? nWords - 1 : nWords;
#if NDSCORE_AVX512 && NDSCORE_VPOPCNT
	{
		__m512i vacc = _mm512_setzero_si512();
		for (; w + 8 <= fullWords; w += 8) {
			__m512i xa = _mm512_loadu_si512(a + w);
			__m512i t;
			if (op == NDScoreOp::L2Self)
				t = xa;
			else {
				__m512i xb = _mm512_loadu_si512(b + w);
				t = (op == NDScoreOp::Dot) ? _mm512_and_si512(xa, xb) : _mm512_xor_si512(xa, xb);
			}
			vacc = _mm512_add_epi64(vacc, _mm512_popcnt_epi64(t));
		}
		acc += (int64_t)_mm512_reduce_add_epi64(vacc);
	}
#elif NDSCORE_NEON
	{
		uint32x4_t vacc = vdupq_n_u32(0);
		for (; w + 2 <= fullWords; w += 2) {
			uint8x16_t xa = vld1q_u8((const uint8_t*)(a + w));
			uint8x16_t t;
			if (op == NDScoreOp::L2Self)
				t = xa;
			else {
				uint8x16_t xb = vld1q_u8((const uint8_t*)(b + w));
				t = (op == NDScoreOp::Dot) ? vandq_u8(xa, xb) : veorq_u8(xa, xb);
			}
			uint8x16_t c = vcntq_u8(t);
#if defined(__aarch64__)
			acc += (int64_t)vaddvq_u8(c);
#else
			uint16x8_t s16 = vpaddlq_u8(c);
			uint32x4_t s32 = vpaddlq_u16(s16);
			vacc = vaddq_u32(vacc, s32);
#endif
		}
#if !defined(__aarch64__)
		uint64x2_t s64 = vpaddlq_u32(vacc);
		acc += (int64_t)(vgetq_lane_u64(s64, 0) + vgetq_lane_u64(s64, 1));
#endif
	}
#endif
	// Scalar popcnt remainder / default (already 64-wide; 4-acc for ILP).
	{
		int64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
		for (; w + 4 <= fullWords; w += 4) {
			uint64_t t0, t1, t2, t3;
			if (op == NDScoreOp::L2Self) {
				t0 = a[w];
				t1 = a[w + 1];
				t2 = a[w + 2];
				t3 = a[w + 3];
			} else if (op == NDScoreOp::Dot) {
				t0 = a[w] & b[w];
				t1 = a[w + 1] & b[w + 1];
				t2 = a[w + 2] & b[w + 2];
				t3 = a[w + 3] & b[w + 3];
			} else {
				t0 = a[w] ^ b[w];
				t1 = a[w + 1] ^ b[w + 1];
				t2 = a[w + 2] ^ b[w + 2];
				t3 = a[w + 3] ^ b[w + 3];
			}
			s0 += (int64_t)popcnt64(t0);
			s1 += (int64_t)popcnt64(t1);
			s2 += (int64_t)popcnt64(t2);
			s3 += (int64_t)popcnt64(t3);
		}
		acc += (s0 + s1) + (s2 + s3);
		for (; w < nWords; ++w) {
			uint64_t t = (op == NDScoreOp::L2Self) ? a[w]
			             : (op == NDScoreOp::Dot)  ? (a[w] & b[w])
			                                       : (a[w] ^ b[w]);
			if (w + 1 == nWords && remBits)
				t &= (uint64_t(1) << remBits) - 1u;
			acc += (int64_t)popcnt64(t);
		}
	}
	return acc;
}

int64_t ndscore_binary_i64(const uint64_t* a, size_t oa, const uint64_t* b, size_t ob,
                           size_t n, NDScoreOp op) {
	if (n == 0)
		return 0;
	if ((oa % 64) != 0 || (op != NDScoreOp::L2Self && (ob % 64) != 0))
		return binary_scalar_bits(a, oa, b, ob, n, op);
	const uint64_t* aw = a + oa / 64;
	const uint64_t* bw = (op == NDScoreOp::L2Self) ? aw : (b + ob / 64);
	const size_t nWords = (n + 63) / 64;
	const size_t rem = n % 64;
	return binary_words_pop(aw, bw, nWords, rem, op);
}
