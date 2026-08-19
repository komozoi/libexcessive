// INTERNAL — packed first-index min/max for contiguous argmin / argmax.
// Include only from NDArray.cpp. Ties keep the first index (strict < / >).

#ifndef LIBEXCESSIVE_SRC_NDARRAY_ARGEXT_H
#define LIBEXCESSIVE_SRC_NDARRAY_ARGEXT_H

#include <climits>
#include <cstddef>
#include <cstdint>

#if defined(_MSC_VER)
#define NDAE_RESTRICT __restrict
#else
#define NDAE_RESTRICT __restrict__
#endif

#if defined(__AVX512F__)
#define NDAE_AVX512 1
#include <immintrin.h>
#else
#define NDAE_AVX512 0
#endif

#if defined(__AVX512BW__)
#define NDAE_AVX512BW 1
#else
#define NDAE_AVX512BW 0
#endif

static unsigned ndaeCtz64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
	return (unsigned)__builtin_ctzll(x);
#else
	unsigned n = 0;
	while ((x & 1ULL) == 0) {
		x >>= 1;
		++n;
	}
	return n;
#endif
}

template <typename T>
static size_t ndaeDense(const T* NDAE_RESTRICT p, size_t n, bool wantMin) {
	T b0 = p[0];
	T b1 = p[0];
	T b2 = p[0];
	T b3 = p[0];
	size_t i0 = 0;
	size_t i1 = 0;
	size_t i2 = 0;
	size_t i3 = 0;
	size_t i = 1;
	if (n >= 4) {
		b0 = p[0];
		b1 = p[1];
		b2 = p[2];
		b3 = p[3];
		i0 = 0;
		i1 = 1;
		i2 = 2;
		i3 = 3;
		i = 4;
		if (wantMin) {
			for (; i + 3 < n; i += 4) {
				if (p[i] < b0) {
					b0 = p[i];
					i0 = i;
				}
				if (p[i + 1] < b1) {
					b1 = p[i + 1];
					i1 = i + 1;
				}
				if (p[i + 2] < b2) {
					b2 = p[i + 2];
					i2 = i + 2;
				}
				if (p[i + 3] < b3) {
					b3 = p[i + 3];
					i3 = i + 3;
				}
			}
		} else {
			for (; i + 3 < n; i += 4) {
				if (p[i] > b0) {
					b0 = p[i];
					i0 = i;
				}
				if (p[i + 1] > b1) {
					b1 = p[i + 1];
					i1 = i + 1;
				}
				if (p[i + 2] > b2) {
					b2 = p[i + 2];
					i2 = i + 2;
				}
				if (p[i + 3] > b3) {
					b3 = p[i + 3];
					i3 = i + 3;
				}
			}
		}
	}
	T best = b0;
	size_t bi = i0;
	if (wantMin ? b1 < best : b1 > best) {
		best = b1;
		bi = i1;
	} else if (b1 == best && i1 < bi) {
		bi = i1;
	}
	if (wantMin ? b2 < best : b2 > best) {
		best = b2;
		bi = i2;
	} else if (b2 == best && i2 < bi) {
		bi = i2;
	}
	if (wantMin ? b3 < best : b3 > best) {
		best = b3;
		bi = i3;
	} else if (b3 == best && i3 < bi) {
		bi = i3;
	}
	for (; i < n; ++i) {
		if (wantMin ? p[i] < best : p[i] > best) {
			best = p[i];
			bi = i;
		}
	}
	return bi;
}

#if NDAE_AVX512
static __m512i ndaeIdx16(void) {
	return _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}

static size_t ndaeHoriz16f(const float* bv, const int32_t* bi, bool wantMin) {
	size_t best = (size_t)bi[0];
	float val = bv[0];
	for (int k = 1; k < 16; ++k) {
		if (wantMin ? bv[k] < val : bv[k] > val) {
			val = bv[k];
			best = (size_t)bi[k];
		} else if (bv[k] == val && (size_t)bi[k] < best) {
			best = (size_t)bi[k];
		}
	}
	return best;
}

static size_t ndaeF32Avx(const float* NDAE_RESTRICT p, size_t n, bool wantMin) {
	if (n > (size_t)INT32_MAX || n < 16)
		return ndaeDense(p, n, wantMin);
	const __m512i lane16 = ndaeIdx16();
	__m512 bestv0 = _mm512_loadu_ps(p);
	__m512i besti0 = lane16;
	__m512 bestv1 = bestv0;
	__m512i besti1 = besti0;
	size_t i = 16;
	if (n >= 32) {
		bestv1 = _mm512_loadu_ps(p + 16);
		besti1 = _mm512_add_epi32(_mm512_set1_epi32(16), lane16);
		i = 32;
		for (; i + 32 <= n; i += 32) {
			__m512 v0 = _mm512_loadu_ps(p + i);
			__m512 v1 = _mm512_loadu_ps(p + i + 16);
			__m512i idx0 = _mm512_add_epi32(_mm512_set1_epi32((int32_t)i), lane16);
			__m512i idx1 = _mm512_add_epi32(_mm512_set1_epi32((int32_t)(i + 16)), lane16);
			__mmask16 m0 = wantMin
				? _mm512_cmp_ps_mask(v0, bestv0, _CMP_LT_OQ)
				: _mm512_cmp_ps_mask(v0, bestv0, _CMP_GT_OQ);
			__mmask16 m1 = wantMin
				? _mm512_cmp_ps_mask(v1, bestv1, _CMP_LT_OQ)
				: _mm512_cmp_ps_mask(v1, bestv1, _CMP_GT_OQ);
			bestv0 = _mm512_mask_blend_ps(m0, bestv0, v0);
			besti0 = _mm512_mask_blend_epi32(m0, besti0, idx0);
			bestv1 = _mm512_mask_blend_ps(m1, bestv1, v1);
			besti1 = _mm512_mask_blend_epi32(m1, besti1, idx1);
		}
	}
	__mmask16 better = wantMin
		? _mm512_cmp_ps_mask(bestv1, bestv0, _CMP_LT_OQ)
		: _mm512_cmp_ps_mask(bestv1, bestv0, _CMP_GT_OQ);
	__mmask16 eq = _mm512_cmp_ps_mask(bestv1, bestv0, _CMP_EQ_OQ);
	__mmask16 earlier = _mm512_cmplt_epi32_mask(besti1, besti0);
	__mmask16 mm = better | (eq & earlier);
	__m512 bestv = _mm512_mask_blend_ps(mm, bestv0, bestv1);
	__m512i besti = _mm512_mask_blend_epi32(mm, besti0, besti1);
	alignas(64) float bv[16];
	alignas(64) int32_t bi[16];
	_mm512_store_ps(bv, bestv);
	_mm512_store_si512((__m512i*)bi, besti);
	size_t best = ndaeHoriz16f(bv, bi, wantMin);
	float val = p[best];
	for (; i < n; ++i) {
		if (wantMin ? p[i] < val : p[i] > val) {
			val = p[i];
			best = i;
		}
	}
	return best;
}

static size_t ndaeF64Avx(const double* NDAE_RESTRICT p, size_t n, bool wantMin) {
	if (n > (size_t)INT32_MAX || n < 8)
		return ndaeDense(p, n, wantMin);
	const __m512i lane8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
	__m512d bestv = _mm512_loadu_pd(p);
	__m512i besti = lane8;
	size_t i = 8;
	for (; i + 8 <= n; i += 8) {
		__m512d v = _mm512_loadu_pd(p + i);
		__m512i idx = _mm512_add_epi64(_mm512_set1_epi64((long long)i), lane8);
		__mmask8 m = wantMin
			? _mm512_cmp_pd_mask(v, bestv, _CMP_LT_OQ)
			: _mm512_cmp_pd_mask(v, bestv, _CMP_GT_OQ);
		bestv = _mm512_mask_blend_pd(m, bestv, v);
		besti = _mm512_mask_blend_epi64(m, besti, idx);
	}
	alignas(64) double bv[8];
	alignas(64) int64_t bi[8];
	_mm512_store_pd(bv, bestv);
	_mm512_store_si512((__m512i*)bi, besti);
	size_t best = (size_t)bi[0];
	double val = bv[0];
	for (int k = 1; k < 8; ++k) {
		if (wantMin ? bv[k] < val : bv[k] > val) {
			val = bv[k];
			best = (size_t)bi[k];
		} else if (bv[k] == val && (size_t)bi[k] < best) {
			best = (size_t)bi[k];
		}
	}
	val = p[best];
	for (; i < n; ++i) {
		if (wantMin ? p[i] < val : p[i] > val) {
			val = p[i];
			best = i;
		}
	}
	return best;
}

static size_t ndaeI32Avx(const int32_t* NDAE_RESTRICT p, size_t n, bool wantMin) {
	if (n > (size_t)INT32_MAX || n < 16)
		return ndaeDense(p, n, wantMin);
	const __m512i lane16 = ndaeIdx16();
	__m512i bestv = _mm512_loadu_si512(p);
	__m512i besti = lane16;
	size_t i = 16;
	for (; i + 16 <= n; i += 16) {
		__m512i v = _mm512_loadu_si512(p + i);
		__m512i idx = _mm512_add_epi32(_mm512_set1_epi32((int32_t)i), lane16);
		__mmask16 m = wantMin
			? _mm512_cmplt_epi32_mask(v, bestv)
			: _mm512_cmpgt_epi32_mask(v, bestv);
		bestv = _mm512_mask_blend_epi32(m, bestv, v);
		besti = _mm512_mask_blend_epi32(m, besti, idx);
	}
	alignas(64) int32_t bv[16];
	alignas(64) int32_t bi[16];
	_mm512_store_si512((__m512i*)bv, bestv);
	_mm512_store_si512((__m512i*)bi, besti);
	size_t best = (size_t)bi[0];
	int32_t val = bv[0];
	for (int k = 1; k < 16; ++k) {
		if (wantMin ? bv[k] < val : bv[k] > val) {
			val = bv[k];
			best = (size_t)bi[k];
		} else if (bv[k] == val && (size_t)bi[k] < best) {
			best = (size_t)bi[k];
		}
	}
	val = p[best];
	for (; i < n; ++i) {
		if (wantMin ? p[i] < val : p[i] > val) {
			val = p[i];
			best = i;
		}
	}
	return best;
}
#endif

#if NDAE_AVX512BW
static int8_t ndaeHMinMaxI8(__m512i v, bool wantMin) {
	__m256i a = _mm512_extracti64x4_epi64(v, 0);
	__m256i b = _mm512_extracti64x4_epi64(v, 1);
	__m256i m = wantMin ? _mm256_min_epi8(a, b) : _mm256_max_epi8(a, b);
	__m128i c = _mm256_castsi256_si128(m);
	__m128i d = _mm256_extracti128_si256(m, 1);
	__m128i n = wantMin ? _mm_min_epi8(c, d) : _mm_max_epi8(c, d);
	n = wantMin ? _mm_min_epi8(n, _mm_srli_si128(n, 8)) : _mm_max_epi8(n, _mm_srli_si128(n, 8));
	n = wantMin ? _mm_min_epi8(n, _mm_srli_si128(n, 4)) : _mm_max_epi8(n, _mm_srli_si128(n, 4));
	n = wantMin ? _mm_min_epi8(n, _mm_srli_si128(n, 2)) : _mm_max_epi8(n, _mm_srli_si128(n, 2));
	n = wantMin ? _mm_min_epi8(n, _mm_srli_si128(n, 1)) : _mm_max_epi8(n, _mm_srli_si128(n, 1));
	return (int8_t)_mm_cvtsi128_si32(n);
}

static uint8_t ndaeHMinMaxU8(__m512i v, bool wantMin) {
	__m256i a = _mm512_extracti64x4_epi64(v, 0);
	__m256i b = _mm512_extracti64x4_epi64(v, 1);
	__m256i m = wantMin ? _mm256_min_epu8(a, b) : _mm256_max_epu8(a, b);
	__m128i c = _mm256_castsi256_si128(m);
	__m128i d = _mm256_extracti128_si256(m, 1);
	__m128i n = wantMin ? _mm_min_epu8(c, d) : _mm_max_epu8(c, d);
	n = wantMin ? _mm_min_epu8(n, _mm_srli_si128(n, 8)) : _mm_max_epu8(n, _mm_srli_si128(n, 8));
	n = wantMin ? _mm_min_epu8(n, _mm_srli_si128(n, 4)) : _mm_max_epu8(n, _mm_srli_si128(n, 4));
	n = wantMin ? _mm_min_epu8(n, _mm_srli_si128(n, 2)) : _mm_max_epu8(n, _mm_srli_si128(n, 2));
	n = wantMin ? _mm_min_epu8(n, _mm_srli_si128(n, 1)) : _mm_max_epu8(n, _mm_srli_si128(n, 1));
	return (uint8_t)_mm_cvtsi128_si32(n);
}

static size_t ndaeI8Avx(const int8_t* NDAE_RESTRICT p, size_t n, bool wantMin) {
	if (n < 64)
		return ndaeDense(p, n, wantMin);
	int8_t best = p[0];
	size_t bestI = 0;
	size_t i = 0;
	for (; i + 64 <= n; i += 64) {
		__m512i v = _mm512_loadu_si512(p + i);
		int8_t chunk = ndaeHMinMaxI8(v, wantMin);
		if (wantMin ? chunk < best : chunk > best) {
			__mmask64 eq = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(chunk));
			best = chunk;
			bestI = i + (size_t)ndaeCtz64((uint64_t)eq);
		}
	}
	for (; i < n; ++i) {
		if (wantMin ? p[i] < best : p[i] > best) {
			best = p[i];
			bestI = i;
		}
	}
	return bestI;
}

static size_t ndaeU8Avx(const uint8_t* NDAE_RESTRICT p, size_t n, bool wantMin) {
	if (n < 64)
		return ndaeDense(p, n, wantMin);
	uint8_t best = p[0];
	size_t bestI = 0;
	size_t i = 0;
	for (; i + 64 <= n; i += 64) {
		__m512i v = _mm512_loadu_si512(p + i);
		uint8_t chunk = ndaeHMinMaxU8(v, wantMin);
		if (wantMin ? chunk < best : chunk > best) {
			__mmask64 eq = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8((int8_t)chunk));
			best = chunk;
			bestI = i + (size_t)ndaeCtz64((uint64_t)eq);
		}
	}
	for (; i < n; ++i) {
		if (wantMin ? p[i] < best : p[i] > best) {
			best = p[i];
			bestI = i;
		}
	}
	return bestI;
}
#endif

static size_t ndaeHalf(NDArrayType t, const uint16_t* NDAE_RESTRICT p, size_t n, bool wantMin) {
	float best = ndarray_half::load(t, p[0]);
	size_t bi = 0;
	size_t i = 1;
	if (n >= 4) {
		float b0 = ndarray_half::load(t, p[0]);
		float b1 = ndarray_half::load(t, p[1]);
		float b2 = ndarray_half::load(t, p[2]);
		float b3 = ndarray_half::load(t, p[3]);
		size_t i0 = 0;
		size_t i1 = 1;
		size_t i2 = 2;
		size_t i3 = 3;
		i = 4;
		if (wantMin) {
			for (; i + 3 < n; i += 4) {
				float v0 = ndarray_half::load(t, p[i]);
				float v1 = ndarray_half::load(t, p[i + 1]);
				float v2 = ndarray_half::load(t, p[i + 2]);
				float v3 = ndarray_half::load(t, p[i + 3]);
				if (v0 < b0) {
					b0 = v0;
					i0 = i;
				}
				if (v1 < b1) {
					b1 = v1;
					i1 = i + 1;
				}
				if (v2 < b2) {
					b2 = v2;
					i2 = i + 2;
				}
				if (v3 < b3) {
					b3 = v3;
					i3 = i + 3;
				}
			}
		} else {
			for (; i + 3 < n; i += 4) {
				float v0 = ndarray_half::load(t, p[i]);
				float v1 = ndarray_half::load(t, p[i + 1]);
				float v2 = ndarray_half::load(t, p[i + 2]);
				float v3 = ndarray_half::load(t, p[i + 3]);
				if (v0 > b0) {
					b0 = v0;
					i0 = i;
				}
				if (v1 > b1) {
					b1 = v1;
					i1 = i + 1;
				}
				if (v2 > b2) {
					b2 = v2;
					i2 = i + 2;
				}
				if (v3 > b3) {
					b3 = v3;
					i3 = i + 3;
				}
			}
		}
		best = b0;
		bi = i0;
		if (wantMin ? b1 < best : b1 > best) {
			best = b1;
			bi = i1;
		} else if (b1 == best && i1 < bi) {
			bi = i1;
		}
		if (wantMin ? b2 < best : b2 > best) {
			best = b2;
			bi = i2;
		} else if (b2 == best && i2 < bi) {
			bi = i2;
		}
		if (wantMin ? b3 < best : b3 > best) {
			best = b3;
			bi = i3;
		} else if (b3 == best && i3 < bi) {
			bi = i3;
		}
	}
	for (; i < n; ++i) {
		float v = ndarray_half::load(t, p[i]);
		if (wantMin ? v < best : v > best) {
			best = v;
			bi = i;
		}
	}
	return bi;
}

static size_t ndaeInt3(const uint64_t* words, size_t start, size_t n, bool wantMin) {
	int best = int3_getSigned(words, start);
	size_t bi = 0;
	const size_t end = start + n;
	size_t e = start + 1;
	while (e < end && (e % int3_kLanesPerWord) != 0) {
		int v = int3_getSigned(words, e);
		if (wantMin ? v < best : v > best) {
			best = v;
			bi = e - start;
		}
		++e;
	}
	while (e + int3_kLanesPerWord <= end) {
		const uint64_t w = words[e / int3_kLanesPerWord];
		for (size_t lane = 0; lane < int3_kLanesPerWord; ++lane) {
			int v = int3_decode((uint8_t)((w >> (lane * 4)) & 7u));
			if (wantMin ? v < best : v > best) {
				best = v;
				bi = (e + lane) - start;
			}
		}
		e += int3_kLanesPerWord;
	}
	for (; e < end; ++e) {
		int v = int3_getSigned(words, e);
		if (wantMin ? v < best : v > best) {
			best = v;
			bi = e - start;
		}
	}
	return bi;
}

static uint64_t ndaeBitKeep(size_t lo, size_t hi) {
	// Bits [lo, hi) in one 64-bit word. lo/hi are 0..64.
	const uint64_t hiMask = (hi >= 64) ? ~0ULL : ((1ULL << hi) - 1ULL);
	const uint64_t loMask = (lo == 0) ? 0ULL : ((1ULL << lo) - 1ULL);
	return hiMask & ~loMask;
}

static size_t ndaeBinary(const uint64_t* words, size_t start, size_t n, bool wantMin) {
	const size_t end = start + n;
	size_t e = start;
	while (e < end) {
		const size_t w = e >> 6;
		const size_t lo = e & 63;
		const size_t wordEnd = (w + 1) << 6;
		const size_t hi = end < wordEnd ? end : wordEnd;
		const uint64_t keep = ndaeBitKeep(lo, hi & 63 ? (hi & 63) : 64);
		const uint64_t bits = words[w];
		const uint64_t hit = wantMin ? ((~bits) & keep) : (bits & keep);
		if (hit) {
			const unsigned tz = ndaeCtz64(hit);
			return (w << 6) + tz - start;
		}
		e = hi;
	}
	return 0;
}

/** Scan n elements starting at element index startE. False if type is unsupported. */
static bool packedArgExt(NDArrayType type, const void* data, size_t startE, size_t n,
                         bool wantMin, size_t* outIdx) {
	if (!data || n == 0 || !outIdx)
		return false;
	switch (type) {
		case F32:
#if NDAE_AVX512
			*outIdx = ndaeF32Avx((const float*)data + startE, n, wantMin);
#else
			*outIdx = ndaeDense((const float*)data + startE, n, wantMin);
#endif
			return true;
		case F64:
#if NDAE_AVX512
			*outIdx = ndaeF64Avx((const double*)data + startE, n, wantMin);
#else
			*outIdx = ndaeDense((const double*)data + startE, n, wantMin);
#endif
			return true;
		case F16:
		case BF16:
			*outIdx = ndaeHalf(type, (const uint16_t*)data + startE, n, wantMin);
			return true;
		case INT8:
#if NDAE_AVX512BW
			*outIdx = ndaeI8Avx((const int8_t*)data + startE, n, wantMin);
#else
			*outIdx = ndaeDense((const int8_t*)data + startE, n, wantMin);
#endif
			return true;
		case UINT8:
#if NDAE_AVX512BW
			*outIdx = ndaeU8Avx((const uint8_t*)data + startE, n, wantMin);
#else
			*outIdx = ndaeDense((const uint8_t*)data + startE, n, wantMin);
#endif
			return true;
		case INT32:
#if NDAE_AVX512
			*outIdx = ndaeI32Avx((const int32_t*)data + startE, n, wantMin);
#else
			*outIdx = ndaeDense((const int32_t*)data + startE, n, wantMin);
#endif
			return true;
		case INT64:
			*outIdx = ndaeDense((const int64_t*)data + startE, n, wantMin);
			return true;
		case INT3:
			*outIdx = ndaeInt3((const uint64_t*)data, startE, n, wantMin);
			return true;
		case BINARY:
			*outIdx = ndaeBinary((const uint64_t*)data, startE, n, wantMin);
			return true;
		default:
			return false;
	}
}

#endif
