// F16 / BF16 kernels. Prefer hardware convert / half ALU; otherwise
// convert-to-F32 SIMD; portable remainder is restrict + 4-wide auto-vec.

#include "ndarray_half_kernels.h"
#include "NDArray.h"

#include <cmath>
#include <cstring>

// GCC -Waggressive-loop-optimizations: `size_t i + k <= n` can wrap, so a
// leftover `i < nn` tail looks like 2^62 pointer steps. ptrdiff_t add is UB
// on overflow, so the compiler must treat n as in-range. Shape products are.
static inline ptrdiff_t ndhLen(size_t n) {
	return (ptrdiff_t)n;
}

#if defined(_MSC_VER)
#define NDH_RESTRICT __restrict
#else
#define NDH_RESTRICT __restrict__
#endif

#if defined(__AVX512F__)
#define NDH_AVX512 1
#include <immintrin.h>
#else
#define NDH_AVX512 0
#endif

#if defined(__AVX512FP16__)
#define NDH_AVX512FP16 1
#else
#define NDH_AVX512FP16 0
#endif

#if defined(__AVX512BF16__)
#define NDH_AVX512BF16 1
#else
#define NDH_AVX512BF16 0
#endif

#if defined(__AVX2__)
#define NDH_AVX2 1
#if !NDH_AVX512
#include <immintrin.h>
#endif
#else
#define NDH_AVX2 0
#endif

#if defined(__F16C__) || defined(__AVX512F__)
#define NDH_F16C 1
#if !NDH_AVX2 && !NDH_AVX512
#include <immintrin.h>
#endif
#else
#define NDH_F16C 0
#endif

#if defined(__SSE2__)
#define NDH_SSE2 1
#if !NDH_AVX2 && !NDH_AVX512 && !NDH_F16C
#include <emmintrin.h>
#endif
#else
#define NDH_SSE2 0
#endif

#if defined(__FMA__)
#define NDH_FMA 1
#else
#define NDH_FMA 0
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define NDH_NEON 1
#include <arm_neon.h>
#else
#define NDH_NEON 0
#endif

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define NDH_NEON_FP16 1
#else
#define NDH_NEON_FP16 0
#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#define NDH_NEON_BF16 1
#else
#define NDH_NEON_BF16 0
#endif

static const int kCvtRound = 0x08; // _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC when available

static inline float ndhF32Add(float a, float b) { return a + b; }
static inline float ndhF32Sub(float a, float b) { return a - b; }
static inline float ndhF32Mul(float a, float b) { return a * b; }
static inline float ndhF32Div(float a, float b) { return a / b; }

static float (*ndhOpFn(NDHalfArith op))(float, float) {
	switch (op) {
		case NDHalfArith::Add: return ndhF32Add;
		case NDHalfArith::Sub: return ndhF32Sub;
		case NDHalfArith::Mul: return ndhF32Mul;
		case NDHalfArith::Div: return ndhF32Div;
	}
	return ndhF32Add;
}

// ---- portable converters (selects, no data-dependent goto — auto-vec) ------

static inline uint32_t f16BitsToF32Bits(uint16_t h) {
	const uint32_t s = (uint32_t)(h & 0x8000u) << 16;
	const uint32_t e = (h >> 10) & 0x1fu;
	const uint32_t m = h & 0x3ffu;
	const float den_f = (float)m * 5.9604644775390625e-8f;
	uint32_t den;
	std::memcpy(&den, &den_f, 4);
	den |= s;
	const uint32_t norm = s | ((e + 112u) << 23) | (m << 13);
	const uint32_t spec = s | 0x7f800000u | (m << 13);
	return (e == 0u) ? den : ((e == 31u) ? spec : norm);
}

static inline float f16LoadPortable(uint16_t h) {
	uint32_t u = f16BitsToF32Bits(h);
	float f;
	std::memcpy(&f, &u, 4);
	return f;
}

static inline uint16_t f16StorePortable(float f) {
	return ndarray_half::f32ToF16(f);
}

static inline float bf16LoadPortable(uint16_t h) {
	uint32_t u = (uint32_t)h << 16;
	float f;
	std::memcpy(&f, &u, 4);
	return f;
}

static inline uint16_t bf16StorePortable(float f) {
	return ndarray_half::f32ToBf16(f);
}

// ---- F16 → F32 -------------------------------------------------------------

void ndhalf_f16_to_f32(float* NDH_RESTRICT dst, const uint16_t* NDH_RESTRICT src, size_t n) {
	const ptrdiff_t nn = ndhLen(n);
	ptrdiff_t i = 0;
#if NDH_AVX512
	for (; i + 16 <= nn; i += 16) {
		__m256i h = _mm256_loadu_si256((const __m256i*)(src + i));
		_mm512_storeu_ps(dst + i, _mm512_cvtph_ps(h));
	}
#endif
#if NDH_AVX2 && NDH_F16C
	for (; i + 8 <= nn; i += 8) {
		__m128i h = _mm_loadu_si128((const __m128i*)(src + i));
		_mm256_storeu_ps(dst + i, _mm256_cvtph_ps(h));
	}
#elif NDH_F16C
	for (; i + 4 <= nn; i += 4) {
		__m128i h = _mm_loadl_epi64((const __m128i*)(src + i));
		_mm_storeu_ps(dst + i, _mm_cvtph_ps(h));
	}
#endif
#if NDH_NEON_FP16
	for (; i + 4 <= nn; i += 4) {
		float16x4_t h = vld1_f16((const float16_t*)(src + i));
		vst1q_f32(dst + i, vcvt_f32_f16(h));
	}
#endif
	for (; i + 4 <= nn; i += 4) {
		dst[i] = f16LoadPortable(src[i]);
		dst[i + 1] = f16LoadPortable(src[i + 1]);
		dst[i + 2] = f16LoadPortable(src[i + 2]);
		dst[i + 3] = f16LoadPortable(src[i + 3]);
	}
	for (; i < nn; ++i)
		dst[i] = f16LoadPortable(src[i]);
}

void ndhalf_f32_to_f16(uint16_t* NDH_RESTRICT dst, const float* NDH_RESTRICT src, size_t n) {
	const ptrdiff_t nn = ndhLen(n);
	ptrdiff_t i = 0;
#if NDH_AVX512
	for (; i + 16 <= nn; i += 16) {
		__m256i h = _mm512_cvtps_ph(_mm512_loadu_ps(src + i), kCvtRound);
		_mm256_storeu_si256((__m256i*)(dst + i), h);
	}
#endif
#if NDH_AVX2 && NDH_F16C
	for (; i + 8 <= nn; i += 8) {
		__m128i h = _mm256_cvtps_ph(_mm256_loadu_ps(src + i), kCvtRound);
		_mm_storeu_si128((__m128i*)(dst + i), h);
	}
#elif NDH_F16C
	for (; i + 4 <= nn; i += 4) {
		__m128i h = _mm_cvtps_ph(_mm_loadu_ps(src + i), kCvtRound);
		_mm_storel_epi64((__m128i*)(dst + i), h);
	}
#endif
#if NDH_NEON_FP16
	for (; i + 4 <= nn; i += 4) {
		float16x4_t h = vcvt_f16_f32(vld1q_f32(src + i));
		vst1_f16((float16_t*)(dst + i), h);
	}
#endif
	for (; i + 4 <= nn; i += 4) {
		dst[i] = f16StorePortable(src[i]);
		dst[i + 1] = f16StorePortable(src[i + 1]);
		dst[i + 2] = f16StorePortable(src[i + 2]);
		dst[i + 3] = f16StorePortable(src[i + 3]);
	}
	for (; i < nn; ++i)
		dst[i] = f16StorePortable(src[i]);
}

// ---- BF16 ↔ F32 ------------------------------------------------------------

void ndhalf_bf16_to_f32(float* NDH_RESTRICT dst, const uint16_t* NDH_RESTRICT src, size_t n) {
	const ptrdiff_t nn = ndhLen(n);
	ptrdiff_t i = 0;
#if NDH_AVX512BF16
	for (; i + 16 <= nn; i += 16) {
		__m256bh h;
		std::memcpy(&h, src + i, 32);
		_mm512_storeu_ps(dst + i, _mm512_cvtpbh_ps(h));
	}
#endif
#if NDH_AVX2
	for (; i + 8 <= nn; i += 8) {
		__m128i h = _mm_loadu_si128((const __m128i*)(src + i));
		__m256i w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
		_mm256_storeu_ps(dst + i, _mm256_castsi256_ps(w));
	}
#endif
#if NDH_NEON
	for (; i + 4 <= nn; i += 4) {
		uint16x4_t h = vld1_u16(src + i);
		uint32x4_t w = vshll_n_u16(h, 16);
		vst1q_f32(dst + i, vreinterpretq_f32_u32(w));
	}
#endif
	for (; i + 4 <= nn; i += 4) {
		dst[i] = bf16LoadPortable(src[i]);
		dst[i + 1] = bf16LoadPortable(src[i + 1]);
		dst[i + 2] = bf16LoadPortable(src[i + 2]);
		dst[i + 3] = bf16LoadPortable(src[i + 3]);
	}
	for (; i < nn; ++i)
		dst[i] = bf16LoadPortable(src[i]);
}

void ndhalf_f32_to_bf16(uint16_t* NDH_RESTRICT dst, const float* NDH_RESTRICT src, size_t n) {
	const ptrdiff_t nn = ndhLen(n);
	ptrdiff_t i = 0;
#if NDH_AVX512BF16
	for (; i + 16 <= nn; i += 16) {
		__m256bh h = _mm512_cvtneps_pbh(_mm512_loadu_ps(src + i));
		std::memcpy(dst + i, &h, 32);
	}
#endif
#if NDH_AVX2
	for (; i + 8 <= nn; i += 8) {
		__m256i x = _mm256_castps_si256(_mm256_loadu_ps(src + i));
		__m256i absv = _mm256_and_si256(x, _mm256_set1_epi32((int)0x7fffffff));
		__m256i isnan = _mm256_cmpgt_epi32(absv, _mm256_set1_epi32((int)0x7f800000));
		__m256i lsb = _mm256_and_si256(_mm256_srli_epi32(x, 16), _mm256_set1_epi32(1));
		__m256i rnd = _mm256_add_epi32(x, _mm256_add_epi32(_mm256_set1_epi32(0x7fff), lsb));
		__m256i hi = _mm256_srli_epi32(rnd, 16);
		__m256i nanv = _mm256_or_si256(_mm256_srli_epi32(x, 16), _mm256_set1_epi32(0x40));
		hi = _mm256_blendv_epi8(hi, nanv, isnan);
		__m128i lo = _mm256_castsi256_si128(hi);
		__m128i hi128 = _mm256_extracti128_si256(hi, 1);
		_mm_storeu_si128((__m128i*)(dst + i), _mm_packus_epi32(lo, hi128));
	}
#endif
#if NDH_NEON
	for (; i + 4 <= nn; i += 4) {
		uint32x4_t x = vreinterpretq_u32_f32(vld1q_f32(src + i));
		uint32x4_t absv = vandq_u32(x, vdupq_n_u32(0x7fffffffu));
		uint32x4_t isnan = vcgtq_u32(absv, vdupq_n_u32(0x7f800000u));
		uint32x4_t lsb = vandq_u32(vshrq_n_u32(x, 16), vdupq_n_u32(1));
		uint32x4_t rnd = vaddq_u32(x, vaddq_u32(vdupq_n_u32(0x7fffu), lsb));
		uint32x4_t hi = vshrq_n_u32(rnd, 16);
		uint32x4_t nanv = vorrq_u32(vshrq_n_u32(x, 16), vdupq_n_u32(0x40u));
		hi = vbslq_u32(isnan, nanv, hi);
		vst1_u16(dst + i, vmovn_u32(hi));
	}
#endif
	for (; i + 4 <= nn; i += 4) {
		dst[i] = bf16StorePortable(src[i]);
		dst[i + 1] = bf16StorePortable(src[i + 1]);
		dst[i + 2] = bf16StorePortable(src[i + 2]);
		dst[i + 3] = bf16StorePortable(src[i + 3]);
	}
	for (; i < nn; ++i)
		dst[i] = bf16StorePortable(src[i]);
}

// ---- F32 SIMD op helpers ---------------------------------------------------

#if NDH_AVX512
static inline __m512 ndhOp512(__m512 a, __m512 b, NDHalfArith op) {
	switch (op) {
		case NDHalfArith::Add: return _mm512_add_ps(a, b);
		case NDHalfArith::Sub: return _mm512_sub_ps(a, b);
		case NDHalfArith::Mul: return _mm512_mul_ps(a, b);
		case NDHalfArith::Div: return _mm512_div_ps(a, b);
	}
	return a;
}
#endif
#if NDH_AVX2
static inline __m256 ndhOp256(__m256 a, __m256 b, NDHalfArith op) {
	switch (op) {
		case NDHalfArith::Add: return _mm256_add_ps(a, b);
		case NDHalfArith::Sub: return _mm256_sub_ps(a, b);
		case NDHalfArith::Mul: return _mm256_mul_ps(a, b);
		case NDHalfArith::Div: return _mm256_div_ps(a, b);
	}
	return a;
}
#endif
#if NDH_F16C && !NDH_AVX2
static inline __m128 ndhOp128(__m128 a, __m128 b, NDHalfArith op) {
	switch (op) {
		case NDHalfArith::Add: return _mm_add_ps(a, b);
		case NDHalfArith::Sub: return _mm_sub_ps(a, b);
		case NDHalfArith::Mul: return _mm_mul_ps(a, b);
		case NDHalfArith::Div: return _mm_div_ps(a, b);
	}
	return a;
}
#endif
#if NDH_NEON
static inline float32x4_t ndhOpN4(float32x4_t a, float32x4_t b, NDHalfArith op) {
	switch (op) {
		case NDHalfArith::Add: return vaddq_f32(a, b);
		case NDHalfArith::Sub: return vsubq_f32(a, b);
		case NDHalfArith::Mul: return vmulq_f32(a, b);
		case NDHalfArith::Div:
#if defined(__aarch64__)
			return vdivq_f32(a, b);
#else
			{
				float ta[4], tb[4], tr[4];
				vst1q_f32(ta, a);
				vst1q_f32(tb, b);
				tr[0] = ta[0] / tb[0];
				tr[1] = ta[1] / tb[1];
				tr[2] = ta[2] / tb[2];
				tr[3] = ta[3] / tb[3];
				return vld1q_f32(tr);
			}
#endif
	}
	return a;
}
#endif

static void ndhArithF32Buf(float* NDH_RESTRICT d, const float* NDH_RESTRICT a,
                           const float* NDH_RESTRICT b, size_t n, NDHalfArith op) {
	const ptrdiff_t nn = ndhLen(n);
	ptrdiff_t i = 0;
	switch (op) {
		case NDHalfArith::Add:
			for (; i + 4 <= nn; i += 4) {
				d[i] = a[i] + b[i];
				d[i + 1] = a[i + 1] + b[i + 1];
				d[i + 2] = a[i + 2] + b[i + 2];
				d[i + 3] = a[i + 3] + b[i + 3];
			}
			for (; i < nn; ++i) d[i] = a[i] + b[i];
			break;
		case NDHalfArith::Sub:
			for (; i + 4 <= nn; i += 4) {
				d[i] = a[i] - b[i];
				d[i + 1] = a[i + 1] - b[i + 1];
				d[i + 2] = a[i + 2] - b[i + 2];
				d[i + 3] = a[i + 3] - b[i + 3];
			}
			for (; i < nn; ++i) d[i] = a[i] - b[i];
			break;
		case NDHalfArith::Mul:
			for (; i + 4 <= nn; i += 4) {
				d[i] = a[i] * b[i];
				d[i + 1] = a[i + 1] * b[i + 1];
				d[i + 2] = a[i + 2] * b[i + 2];
				d[i + 3] = a[i + 3] * b[i + 3];
			}
			for (; i < nn; ++i) d[i] = a[i] * b[i];
			break;
		case NDHalfArith::Div:
			for (; i + 4 <= nn; i += 4) {
				d[i] = a[i] / b[i];
				d[i + 1] = a[i + 1] / b[i + 1];
				d[i + 2] = a[i + 2] / b[i + 2];
				d[i + 3] = a[i + 3] / b[i + 3];
			}
			for (; i < nn; ++i) d[i] = a[i] / b[i];
			break;
	}
}

// ---- F16 arithmetic --------------------------------------------------------

#if NDH_AVX512FP16
static inline __m512h ndhOpPh(__m512h a, __m512h b, NDHalfArith op) {
	switch (op) {
		case NDHalfArith::Add: return _mm512_add_ph(a, b);
		case NDHalfArith::Sub: return _mm512_sub_ph(a, b);
		case NDHalfArith::Mul: return _mm512_mul_ph(a, b);
		case NDHalfArith::Div: return _mm512_div_ph(a, b);
	}
	return a;
}
#endif

#if NDH_NEON_FP16
static inline float16x8_t ndhOpH8(float16x8_t a, float16x8_t b, NDHalfArith op) {
	switch (op) {
		case NDHalfArith::Add: return vaddq_f16(a, b);
		case NDHalfArith::Sub: return vsubq_f16(a, b);
		case NDHalfArith::Mul: return vmulq_f16(a, b);
		case NDHalfArith::Div: return vdivq_f16(a, b);
	}
	return a;
}
#endif

void ndhalf_arith_f16(uint16_t* NDH_RESTRICT dst, const uint16_t* NDH_RESTRICT a,
                      const uint16_t* NDH_RESTRICT b, size_t n, NDHalfArith op) {
	const ptrdiff_t nn = ndhLen(n);
	ptrdiff_t i = 0;
#if NDH_AVX512FP16
	for (; i + 32 <= nn; i += 32) {
		__m512h xa = _mm512_castsi512_ph(_mm512_loadu_si512(a + i));
		__m512h xb = _mm512_castsi512_ph(_mm512_loadu_si512(b + i));
		_mm512_storeu_si512(dst + i, _mm512_castph_si512(ndhOpPh(xa, xb, op)));
	}
#endif
#if NDH_NEON_FP16
	for (; i + 8 <= nn; i += 8) {
		float16x8_t xa = vld1q_f16((const float16_t*)(a + i));
		float16x8_t xb = vld1q_f16((const float16_t*)(b + i));
		vst1q_f16((float16_t*)(dst + i), ndhOpH8(xa, xb, op));
	}
#endif
#if NDH_AVX512
	for (; i + 16 <= nn; i += 16) {
		__m512 fa = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(a + i)));
		__m512 fb = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(b + i)));
		__m256i h = _mm512_cvtps_ph(ndhOp512(fa, fb, op), kCvtRound);
		_mm256_storeu_si256((__m256i*)(dst + i), h);
	}
#endif
#if NDH_AVX2 && NDH_F16C
	for (; i + 8 <= nn; i += 8) {
		__m256 fa = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(a + i)));
		__m256 fb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + i)));
		_mm_storeu_si128((__m128i*)(dst + i), _mm256_cvtps_ph(ndhOp256(fa, fb, op), kCvtRound));
	}
#elif NDH_F16C
	for (; i + 4 <= nn; i += 4) {
		__m128 fa = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)(a + i)));
		__m128 fb = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)(b + i)));
		_mm_storel_epi64((__m128i*)(dst + i), _mm_cvtps_ph(ndhOp128(fa, fb, op), kCvtRound));
	}
#endif
#if NDH_NEON && !NDH_NEON_FP16
	for (; i + 4 <= nn; i += 4) {
		float ta[4], tb[4], tr[4];
		ndhalf_f16_to_f32(ta, a + i, 4);
		ndhalf_f16_to_f32(tb, b + i, 4);
		vst1q_f32(tr, ndhOpN4(vld1q_f32(ta), vld1q_f32(tb), op));
		ndhalf_f32_to_f16(dst + i, tr, 4);
	}
#endif
	{
		float ta[32], tb[32], tr[32];
		for (; i + 32 <= nn; i += 32) {
			ndhalf_f16_to_f32(ta, a + i, 32);
			ndhalf_f16_to_f32(tb, b + i, 32);
			ndhArithF32Buf(tr, ta, tb, 32, op);
			ndhalf_f32_to_f16(dst + i, tr, 32);
		}
		if (i < nn) {
			const size_t rem = (size_t)(nn - i);
			ndhalf_f16_to_f32(ta, a + i, rem);
			ndhalf_f16_to_f32(tb, b + i, rem);
			ndhArithF32Buf(tr, ta, tb, rem, op);
			ndhalf_f32_to_f16(dst + i, tr, rem);
		}
	}
}

void ndhalf_arith_bf16(uint16_t* NDH_RESTRICT dst, const uint16_t* NDH_RESTRICT a,
                       const uint16_t* NDH_RESTRICT b, size_t n, NDHalfArith op) {
	const ptrdiff_t nn = ndhLen(n);
	ptrdiff_t i = 0;
#if NDH_AVX512BF16
	for (; i + 16 <= nn; i += 16) {
		__m256bh ha, hb;
		std::memcpy(&ha, a + i, 32);
		std::memcpy(&hb, b + i, 32);
		__m512 fa = _mm512_cvtpbh_ps(ha);
		__m512 fb = _mm512_cvtpbh_ps(hb);
		__m256bh hr = _mm512_cvtneps_pbh(ndhOp512(fa, fb, op));
		std::memcpy(dst + i, &hr, 32);
	}
#endif
#if NDH_AVX2
	for (; i + 8 <= nn; i += 8) {
		__m256 fa = _mm256_castsi256_ps(_mm256_slli_epi32(
			_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(a + i))), 16));
		__m256 fb = _mm256_castsi256_ps(_mm256_slli_epi32(
			_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(b + i))), 16));
		float tr[8];
		_mm256_storeu_ps(tr, ndhOp256(fa, fb, op));
		ndhalf_f32_to_bf16(dst + i, tr, 8);
	}
#endif
#if NDH_NEON
	for (; i + 4 <= nn; i += 4) {
		float32x4_t fa = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(a + i), 16));
		float32x4_t fb = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(b + i), 16));
		float tr[4];
		vst1q_f32(tr, ndhOpN4(fa, fb, op));
		ndhalf_f32_to_bf16(dst + i, tr, 4);
	}
#endif
	{
		float ta[32], tb[32], tr[32];
		for (; i + 32 <= nn; i += 32) {
			ndhalf_bf16_to_f32(ta, a + i, 32);
			ndhalf_bf16_to_f32(tb, b + i, 32);
			ndhArithF32Buf(tr, ta, tb, 32, op);
			ndhalf_f32_to_bf16(dst + i, tr, 32);
		}
		if (i < nn) {
			const size_t rem = (size_t)(nn - i);
			ndhalf_bf16_to_f32(ta, a + i, rem);
			ndhalf_bf16_to_f32(tb, b + i, rem);
			ndhArithF32Buf(tr, ta, tb, rem, op);
			ndhalf_f32_to_bf16(dst + i, tr, rem);
		}
	}
}

static void ndhArithScalarF16(uint16_t* dst, const uint16_t* a, float s, size_t n, NDHalfArith op,
                              bool bf) {
	float ta[32], tr[32];
	const ptrdiff_t nn = ndhLen(n);
	ptrdiff_t i = 0;
	for (; i + 32 <= nn; i += 32) {
		if (bf) ndhalf_bf16_to_f32(ta, a + i, 32);
		else ndhalf_f16_to_f32(ta, a + i, 32);
		switch (op) {
			case NDHalfArith::Add:
				for (int j = 0; j < 32; ++j) tr[j] = ta[j] + s;
				break;
			case NDHalfArith::Sub:
				for (int j = 0; j < 32; ++j) tr[j] = ta[j] - s;
				break;
			case NDHalfArith::Mul:
				for (int j = 0; j < 32; ++j) tr[j] = ta[j] * s;
				break;
			case NDHalfArith::Div:
				for (int j = 0; j < 32; ++j) tr[j] = ta[j] / s;
				break;
		}
		if (bf) ndhalf_f32_to_bf16(dst + i, tr, 32);
		else ndhalf_f32_to_f16(dst + i, tr, 32);
	}
	if (i < nn) {
		const size_t rem = (size_t)(nn - i);
		if (bf) ndhalf_bf16_to_f32(ta, a + i, rem);
		else ndhalf_f16_to_f32(ta, a + i, rem);
		switch (op) {
			case NDHalfArith::Add:
				for (size_t j = 0; j < rem; ++j) tr[j] = ta[j] + s;
				break;
			case NDHalfArith::Sub:
				for (size_t j = 0; j < rem; ++j) tr[j] = ta[j] - s;
				break;
			case NDHalfArith::Mul:
				for (size_t j = 0; j < rem; ++j) tr[j] = ta[j] * s;
				break;
			case NDHalfArith::Div:
				for (size_t j = 0; j < rem; ++j) tr[j] = ta[j] / s;
				break;
		}
		if (bf) ndhalf_f32_to_bf16(dst + i, tr, rem);
		else ndhalf_f32_to_f16(dst + i, tr, rem);
	}
	(void)ndhOpFn(op);
}

void ndhalf_arith_f16_scalar(uint16_t* dst, const uint16_t* a, float s, size_t n, NDHalfArith op) {
	ndhArithScalarF16(dst, a, s, n, op, false);
}
void ndhalf_arith_bf16_scalar(uint16_t* dst, const uint16_t* a, float s, size_t n, NDHalfArith op) {
	ndhArithScalarF16(dst, a, s, n, op, true);
}

static void ndhMinMax(uint16_t* dst, const uint16_t* a, const uint16_t* b, size_t n, bool isMin, bool bf) {
	float ta[32], tb[32], tr[32];
	const ptrdiff_t nn = ndhLen(n);
	ptrdiff_t i = 0;
	for (; i + 32 <= nn; i += 32) {
		if (bf) {
			ndhalf_bf16_to_f32(ta, a + i, 32);
			ndhalf_bf16_to_f32(tb, b + i, 32);
		} else {
			ndhalf_f16_to_f32(ta, a + i, 32);
			ndhalf_f16_to_f32(tb, b + i, 32);
		}
		if (isMin) {
			for (int j = 0; j < 32; ++j) tr[j] = ta[j] < tb[j] ? ta[j] : tb[j];
		} else {
			for (int j = 0; j < 32; ++j) tr[j] = ta[j] > tb[j] ? ta[j] : tb[j];
		}
		if (bf) ndhalf_f32_to_bf16(dst + i, tr, 32);
		else ndhalf_f32_to_f16(dst + i, tr, 32);
	}
	if (i < nn) {
		const size_t rem = (size_t)(nn - i);
		if (bf) {
			ndhalf_bf16_to_f32(ta, a + i, rem);
			ndhalf_bf16_to_f32(tb, b + i, rem);
		} else {
			ndhalf_f16_to_f32(ta, a + i, rem);
			ndhalf_f16_to_f32(tb, b + i, rem);
		}
		if (isMin) {
			for (size_t j = 0; j < rem; ++j) tr[j] = ta[j] < tb[j] ? ta[j] : tb[j];
		} else {
			for (size_t j = 0; j < rem; ++j) tr[j] = ta[j] > tb[j] ? ta[j] : tb[j];
		}
		if (bf) ndhalf_f32_to_bf16(dst + i, tr, rem);
		else ndhalf_f32_to_f16(dst + i, tr, rem);
	}
}

void ndhalf_min_f16(uint16_t* d, const uint16_t* a, const uint16_t* b, size_t n) {
	ndhMinMax(d, a, b, n, true, false);
}
void ndhalf_max_f16(uint16_t* d, const uint16_t* a, const uint16_t* b, size_t n) {
	ndhMinMax(d, a, b, n, false, false);
}
void ndhalf_min_bf16(uint16_t* d, const uint16_t* a, const uint16_t* b, size_t n) {
	ndhMinMax(d, a, b, n, true, true);
}
void ndhalf_max_bf16(uint16_t* d, const uint16_t* a, const uint16_t* b, size_t n) {
	ndhMinMax(d, a, b, n, false, true);
}

// ---- reductions / scores (F32 accumulator) ---------------------------------

#if NDH_AVX2
static inline __m256 ndhFmadd256(__m256 a, __m256 b, __m256 c) {
#if NDH_FMA
	return _mm256_fmadd_ps(a, b, c);
#else
	return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}
#endif

#if NDH_AVX2
static inline float ndhHsum256(__m256 v) {
	__m128 lo = _mm256_castps256_ps128(v);
	__m128 hi = _mm256_extractf128_ps(v, 1);
	__m128 s = _mm_add_ps(lo, hi);
	s = _mm_add_ps(s, _mm_movehdup_ps(s));
	s = _mm_add_ss(s, _mm_movehl_ps(s, s));
	return _mm_cvtss_f32(s);
}
#endif

static float ndhalf_reduce_f16(const uint16_t* a, const uint16_t* b, size_t n, int kind) {
	// kind: 0=sum, 1=prod, 2=dot, 3=l2self, 4=l2pair
	const ptrdiff_t nn = ndhLen(n);
	ptrdiff_t i = 0;
#if NDH_AVX512
	__m512 acc0 = (kind == 1) ? _mm512_set1_ps(1.0f) : _mm512_setzero_ps();
	__m512 acc1 = (kind == 1) ? _mm512_set1_ps(1.0f) : _mm512_setzero_ps();
	for (; i + 32 <= nn; i += 32) {
		__m512 x0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(a + i)));
		__m512 x1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(a + i + 16)));
		if (kind == 0) {
			acc0 = _mm512_add_ps(acc0, x0);
			acc1 = _mm512_add_ps(acc1, x1);
		} else if (kind == 1) {
			acc0 = _mm512_mul_ps(acc0, x0);
			acc1 = _mm512_mul_ps(acc1, x1);
		} else if (kind == 2) {
			__m512 y0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(b + i)));
			__m512 y1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(b + i + 16)));
			acc0 = _mm512_fmadd_ps(x0, y0, acc0);
			acc1 = _mm512_fmadd_ps(x1, y1, acc1);
		} else if (kind == 3) {
			acc0 = _mm512_fmadd_ps(x0, x0, acc0);
			acc1 = _mm512_fmadd_ps(x1, x1, acc1);
		} else {
			__m512 y0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(b + i)));
			__m512 y1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(b + i + 16)));
			__m512 d0 = _mm512_sub_ps(x0, y0);
			__m512 d1 = _mm512_sub_ps(x1, y1);
			acc0 = _mm512_fmadd_ps(d0, d0, acc0);
			acc1 = _mm512_fmadd_ps(d1, d1, acc1);
		}
	}
	float s;
	if (kind == 1)
		s = _mm512_reduce_mul_ps(_mm512_mul_ps(acc0, acc1));
	else
		s = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
#elif NDH_AVX2 && NDH_F16C
	__m256 acc0 = (kind == 1) ? _mm256_set1_ps(1.0f) : _mm256_setzero_ps();
	__m256 acc1 = (kind == 1) ? _mm256_set1_ps(1.0f) : _mm256_setzero_ps();
	for (; i + 16 <= nn; i += 16) {
		__m256 x0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(a + i)));
		__m256 x1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(a + i + 8)));
		if (kind == 0) {
			acc0 = _mm256_add_ps(acc0, x0);
			acc1 = _mm256_add_ps(acc1, x1);
		} else if (kind == 1) {
			acc0 = _mm256_mul_ps(acc0, x0);
			acc1 = _mm256_mul_ps(acc1, x1);
		} else if (kind == 2) {
			__m256 y0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + i)));
			__m256 y1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + i + 8)));
			acc0 = ndhFmadd256(x0, y0, acc0);
			acc1 = ndhFmadd256(x1, y1, acc1);
		} else if (kind == 3) {
			acc0 = ndhFmadd256(x0, x0, acc0);
			acc1 = ndhFmadd256(x1, x1, acc1);
		} else {
			__m256 y0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + i)));
			__m256 y1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + i + 8)));
			__m256 d0 = _mm256_sub_ps(x0, y0);
			__m256 d1 = _mm256_sub_ps(x1, y1);
			acc0 = ndhFmadd256(d0, d0, acc0);
			acc1 = ndhFmadd256(d1, d1, acc1);
		}
	}
	float s;
	if (kind == 1) {
		float t[8];
		_mm256_storeu_ps(t, _mm256_mul_ps(acc0, acc1));
		s = t[0] * t[1] * t[2] * t[3] * t[4] * t[5] * t[6] * t[7];
	} else {
		s = ndhHsum256(_mm256_add_ps(acc0, acc1));
	}
#elif NDH_NEON
	float32x4_t acc0 = (kind == 1) ? vdupq_n_f32(1.0f) : vdupq_n_f32(0.0f);
	float32x4_t acc1 = (kind == 1) ? vdupq_n_f32(1.0f) : vdupq_n_f32(0.0f);
	for (; i + 8 <= nn; i += 8) {
		float x0s[4], x1s[4];
		ndhalf_f16_to_f32(x0s, a + i, 4);
		ndhalf_f16_to_f32(x1s, a + i + 4, 4);
		float32x4_t x0 = vld1q_f32(x0s);
		float32x4_t x1 = vld1q_f32(x1s);
		if (kind == 0) {
			acc0 = vaddq_f32(acc0, x0);
			acc1 = vaddq_f32(acc1, x1);
		} else if (kind == 1) {
			acc0 = vmulq_f32(acc0, x0);
			acc1 = vmulq_f32(acc1, x1);
		} else if (kind == 2) {
			float y0s[4], y1s[4];
			ndhalf_f16_to_f32(y0s, b + i, 4);
			ndhalf_f16_to_f32(y1s, b + i + 4, 4);
			acc0 = vmlaq_f32(acc0, x0, vld1q_f32(y0s));
			acc1 = vmlaq_f32(acc1, x1, vld1q_f32(y1s));
		} else if (kind == 3) {
			acc0 = vmlaq_f32(acc0, x0, x0);
			acc1 = vmlaq_f32(acc1, x1, x1);
		} else {
			float y0s[4], y1s[4];
			ndhalf_f16_to_f32(y0s, b + i, 4);
			ndhalf_f16_to_f32(y1s, b + i + 4, 4);
			float32x4_t d0 = vsubq_f32(x0, vld1q_f32(y0s));
			float32x4_t d1 = vsubq_f32(x1, vld1q_f32(y1s));
			acc0 = vmlaq_f32(acc0, d0, d0);
			acc1 = vmlaq_f32(acc1, d1, d1);
		}
	}
	acc0 = vaddq_f32(acc0, acc1);
#if defined(__aarch64__)
	float s = (kind == 1)
		? vgetq_lane_f32(acc0, 0) * vgetq_lane_f32(acc0, 1) *
		  vgetq_lane_f32(acc0, 2) * vgetq_lane_f32(acc0, 3)
		: vaddvq_f32(acc0);
	if (kind == 1) {
		float t[4];
		vst1q_f32(t, acc0);
		s = t[0] * t[1] * t[2] * t[3];
	}
#else
	float t[4];
	vst1q_f32(t, acc0);
	float s = (kind == 1) ? (t[0] * t[1] * t[2] * t[3]) : (t[0] + t[1] + t[2] + t[3]);
#endif
#else
	float s0 = (kind == 1) ? 1.0f : 0.0f;
	float s1 = (kind == 1) ? 1.0f : 0.0f;
	float s2 = (kind == 1) ? 1.0f : 0.0f;
	float s3 = (kind == 1) ? 1.0f : 0.0f;
	for (; i + 4 <= nn; i += 4) {
		float x0 = f16LoadPortable(a[i]);
		float x1 = f16LoadPortable(a[i + 1]);
		float x2 = f16LoadPortable(a[i + 2]);
		float x3 = f16LoadPortable(a[i + 3]);
		if (kind == 0) {
			s0 += x0; s1 += x1; s2 += x2; s3 += x3;
		} else if (kind == 1) {
			s0 *= x0; s1 *= x1; s2 *= x2; s3 *= x3;
		} else if (kind == 2) {
			s0 += x0 * f16LoadPortable(b[i]);
			s1 += x1 * f16LoadPortable(b[i + 1]);
			s2 += x2 * f16LoadPortable(b[i + 2]);
			s3 += x3 * f16LoadPortable(b[i + 3]);
		} else if (kind == 3) {
			s0 += x0 * x0; s1 += x1 * x1; s2 += x2 * x2; s3 += x3 * x3;
		} else {
			float d0 = x0 - f16LoadPortable(b[i]);
			float d1 = x1 - f16LoadPortable(b[i + 1]);
			float d2 = x2 - f16LoadPortable(b[i + 2]);
			float d3 = x3 - f16LoadPortable(b[i + 3]);
			s0 += d0 * d0; s1 += d1 * d1; s2 += d2 * d2; s3 += d3 * d3;
		}
	}
	float s = (kind == 1) ? (s0 * s1 * s2 * s3) : ((s0 + s1) + (s2 + s3));
#endif
	for (; i < nn; ++i) {
		float x = f16LoadPortable(a[i]);
		if (kind == 0) s += x;
		else if (kind == 1) s *= x;
		else if (kind == 2) s += x * f16LoadPortable(b[i]);
		else if (kind == 3) s += x * x;
		else {
			float d = x - f16LoadPortable(b[i]);
			s += d * d;
		}
	}
	return s;
}

static float ndhalf_reduce_bf16(const uint16_t* a, const uint16_t* b, size_t n, int kind) {
	const ptrdiff_t nn = ndhLen(n);
	ptrdiff_t i = 0;
#if NDH_AVX512BF16
	if (kind == 2) {
		__m512 acc = _mm512_setzero_ps();
		for (; i + 32 <= nn; i += 32) {
			__m512bh xa, xb;
			std::memcpy(&xa, a + i, 64);
			std::memcpy(&xb, b + i, 64);
			acc = _mm512_dpbf16_ps(acc, xa, xb);
		}
		float s = _mm512_reduce_add_ps(acc);
		for (; i < nn; ++i)
			s += bf16LoadPortable(a[i]) * bf16LoadPortable(b[i]);
		return s;
	}
#endif
#if NDH_AVX2
	__m256 acc0 = (kind == 1) ? _mm256_set1_ps(1.0f) : _mm256_setzero_ps();
	__m256 acc1 = (kind == 1) ? _mm256_set1_ps(1.0f) : _mm256_setzero_ps();
	for (; i + 16 <= nn; i += 16) {
		__m256 x0 = _mm256_castsi256_ps(_mm256_slli_epi32(
			_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(a + i))), 16));
		__m256 x1 = _mm256_castsi256_ps(_mm256_slli_epi32(
			_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(a + i + 8))), 16));
		if (kind == 0) {
			acc0 = _mm256_add_ps(acc0, x0);
			acc1 = _mm256_add_ps(acc1, x1);
		} else if (kind == 1) {
			acc0 = _mm256_mul_ps(acc0, x0);
			acc1 = _mm256_mul_ps(acc1, x1);
		} else if (kind == 2) {
			__m256 y0 = _mm256_castsi256_ps(_mm256_slli_epi32(
				_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(b + i))), 16));
			__m256 y1 = _mm256_castsi256_ps(_mm256_slli_epi32(
				_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(b + i + 8))), 16));
			acc0 = ndhFmadd256(x0, y0, acc0);
			acc1 = ndhFmadd256(x1, y1, acc1);
		} else if (kind == 3) {
			acc0 = ndhFmadd256(x0, x0, acc0);
			acc1 = ndhFmadd256(x1, x1, acc1);
		} else {
			__m256 y0 = _mm256_castsi256_ps(_mm256_slli_epi32(
				_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(b + i))), 16));
			__m256 y1 = _mm256_castsi256_ps(_mm256_slli_epi32(
				_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(b + i + 8))), 16));
			__m256 d0 = _mm256_sub_ps(x0, y0);
			__m256 d1 = _mm256_sub_ps(x1, y1);
			acc0 = ndhFmadd256(d0, d0, acc0);
			acc1 = ndhFmadd256(d1, d1, acc1);
		}
	}
	float s;
	if (kind == 1) {
		float t[8];
		_mm256_storeu_ps(t, _mm256_mul_ps(acc0, acc1));
		s = t[0] * t[1] * t[2] * t[3] * t[4] * t[5] * t[6] * t[7];
	} else {
		s = ndhHsum256(_mm256_add_ps(acc0, acc1));
	}
#else
	float s0 = (kind == 1) ? 1.0f : 0.0f;
	float s1 = (kind == 1) ? 1.0f : 0.0f;
	float s2 = (kind == 1) ? 1.0f : 0.0f;
	float s3 = (kind == 1) ? 1.0f : 0.0f;
	for (; i + 4 <= nn; i += 4) {
		float x0 = bf16LoadPortable(a[i]);
		float x1 = bf16LoadPortable(a[i + 1]);
		float x2 = bf16LoadPortable(a[i + 2]);
		float x3 = bf16LoadPortable(a[i + 3]);
		if (kind == 0) {
			s0 += x0; s1 += x1; s2 += x2; s3 += x3;
		} else if (kind == 1) {
			s0 *= x0; s1 *= x1; s2 *= x2; s3 *= x3;
		} else if (kind == 2) {
			s0 += x0 * bf16LoadPortable(b[i]);
			s1 += x1 * bf16LoadPortable(b[i + 1]);
			s2 += x2 * bf16LoadPortable(b[i + 2]);
			s3 += x3 * bf16LoadPortable(b[i + 3]);
		} else if (kind == 3) {
			s0 += x0 * x0; s1 += x1 * x1; s2 += x2 * x2; s3 += x3 * x3;
		} else {
			float d0 = x0 - bf16LoadPortable(b[i]);
			float d1 = x1 - bf16LoadPortable(b[i + 1]);
			float d2 = x2 - bf16LoadPortable(b[i + 2]);
			float d3 = x3 - bf16LoadPortable(b[i + 3]);
			s0 += d0 * d0; s1 += d1 * d1; s2 += d2 * d2; s3 += d3 * d3;
		}
	}
	float s = (kind == 1) ? (s0 * s1 * s2 * s3) : ((s0 + s1) + (s2 + s3));
#endif
	for (; i < nn; ++i) {
		float x = bf16LoadPortable(a[i]);
		if (kind == 0) s += x;
		else if (kind == 1) s *= x;
		else if (kind == 2) s += x * bf16LoadPortable(b[i]);
		else if (kind == 3) s += x * x;
		else {
			float d = x - bf16LoadPortable(b[i]);
			s += d * d;
		}
	}
	return s;
}

float ndhalf_sum_f16(const uint16_t* a, size_t n) { return ndhalf_reduce_f16(a, nullptr, n, 0); }
float ndhalf_sum_bf16(const uint16_t* a, size_t n) { return ndhalf_reduce_bf16(a, nullptr, n, 0); }
float ndhalf_prod_f16(const uint16_t* a, size_t n) { return ndhalf_reduce_f16(a, nullptr, n, 1); }
float ndhalf_prod_bf16(const uint16_t* a, size_t n) { return ndhalf_reduce_bf16(a, nullptr, n, 1); }
float ndhalf_dot_f16(const uint16_t* a, const uint16_t* b, size_t n) { return ndhalf_reduce_f16(a, b, n, 2); }
float ndhalf_dot_bf16(const uint16_t* a, const uint16_t* b, size_t n) { return ndhalf_reduce_bf16(a, b, n, 2); }
float ndhalf_l2self_f16(const uint16_t* a, size_t n) { return ndhalf_reduce_f16(a, nullptr, n, 3); }
float ndhalf_l2self_bf16(const uint16_t* a, size_t n) { return ndhalf_reduce_bf16(a, nullptr, n, 3); }
float ndhalf_l2pair_f16(const uint16_t* a, const uint16_t* b, size_t n) { return ndhalf_reduce_f16(a, b, n, 4); }
float ndhalf_l2pair_bf16(const uint16_t* a, const uint16_t* b, size_t n) { return ndhalf_reduce_bf16(a, b, n, 4); }
