// Copyright 2021-2026 Komozoi
// Original Creation Date: 2026-8-4
//
// All rights reserved.
//
// INTERNAL to libexcessive — include only from NDArray.cpp.
// Every symbol is static so nothing is exported from the library object.
//
// INT3: signed 3-bit (-4..3) packed as 4-bit nibbles (pad + 3 value bits),
// 16 values per uint64. Mul/div use 64-entry tables (index = a | (b << 3)).
// AVX-512 VPERMB path when this TU is compiled with the matching ISA flags.

#ifndef LIBEXCESSIVE_SRC_NDARRAY_INT3_H
#define LIBEXCESSIVE_SRC_NDARRAY_INT3_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VBMI__)
#define LIBEXCESSIVE_INT3_AVX512 1
#include <immintrin.h>
#else
#define LIBEXCESSIVE_INT3_AVX512 0
#endif

// ---- constants -------------------------------------------------------------

static constexpr size_t int3_kLanesPerWord = 16;
static constexpr uint64_t int3_kValueMask = 0x7777777777777777ULL;
static constexpr uint64_t int3_kPadMask   = 0x8888888888888888ULL;

// ---- encode / decode / pack ------------------------------------------------

static inline int int3_decode(uint8_t t) {
	t &= 7;
	return (t >= 4) ? (int)t - 8 : (int)t;
}

static inline uint8_t int3_encode(int v) {
	return (uint8_t)(v & 7);
}

static inline size_t int3_bufferBytes(size_t n) {
	return ((n + int3_kLanesPerWord - 1) / int3_kLanesPerWord) * sizeof(uint64_t);
}

static inline size_t int3_wordCount(size_t nElems) {
	return (nElems + int3_kLanesPerWord - 1) / int3_kLanesPerWord;
}

static inline uint8_t int3_get(const uint64_t* words, size_t i) {
	const size_t w = i / int3_kLanesPerWord;
	const unsigned shift = (unsigned)((i % int3_kLanesPerWord) * 4);
	return (uint8_t)((words[w] >> shift) & 7u);
}

static inline void int3_set(uint64_t* words, size_t i, uint8_t pattern3) {
	const size_t w = i / int3_kLanesPerWord;
	const unsigned shift = (unsigned)((i % int3_kLanesPerWord) * 4);
	const uint64_t clear = ~(0xFULL << shift);
	words[w] = (words[w] & clear) | ((uint64_t)(pattern3 & 7u) << shift);
}

static inline int int3_getSigned(const uint64_t* words, size_t i) {
	return int3_decode(int3_get(words, i));
}

static inline void int3_setSigned(uint64_t* words, size_t i, int v) {
	int3_set(words, i, int3_encode(v));
}

static inline uint64_t int3_splatWord(uint8_t pattern3) {
	pattern3 &= 7;
	uint64_t w = 0;
	for (int lane = 0; lane < 16; ++lane)
		w |= (uint64_t)pattern3 << (lane * 4);
	return w;
}

static inline uint64_t int3_addWord(uint64_t a, uint64_t b) {
	return (a + b) & int3_kValueMask;
}

/** Pad bit 0 after sub ⇒ borrow on that nibble. */
static inline uint64_t int3_subWord(uint64_t a, uint64_t b) {
	return ((a | int3_kPadMask) - b) & int3_kValueMask;
}

// ---- tables ----------------------------------------------------------------

// index = a | (b << 3); mul = (a * b) & 7
alignas(64) static const uint8_t int3_kMulTable[64] = {
	// b=0
	0, 0, 0, 0, 0, 0, 0, 0,
	// b=1
	0, 1, 2, 3, 4, 5, 6, 7,
	// b=2
	0, 2, 4, 6, 0, 2, 4, 6,
	// b=3
	0, 3, 6, 1, 4, 7, 2, 5,
	// b=4
	0, 4, 0, 4, 0, 4, 0, 4,
	// b=5
	0, 5, 2, 7, 4, 1, 6, 3,
	// b=6
	0, 6, 4, 2, 0, 6, 4, 2,
	// b=7
	0, 7, 6, 5, 4, 3, 2, 1,
};

// index = a | (b << 3); toward-zero signed; /0 → 0 (callers reject zeros)
alignas(64) static const uint8_t int3_kDivTable[64] = {
	// b=0
	0, 0, 0, 0, 0, 0, 0, 0,
	// b=1
	0, 1, 2, 3, 4, 5, 6, 7,
	// b=2
	0, 0, 1, 1, 6, 7, 7, 0,
	// b=3
	0, 0, 0, 1, 7, 7, 0, 0,
	// b=4
	0, 0, 0, 0, 1, 0, 0, 0,
	// b=5
	0, 0, 0, 7, 1, 1, 0, 0,
	// b=6
	0, 0, 7, 7, 2, 1, 1, 0,
	// b=7
	0, 7, 6, 5, 4, 3, 2, 1,
};

static inline void int3_clearPadding(uint64_t* dst, size_t nElems) {
	const size_t nWords = int3_wordCount(nElems);
	for (size_t i = nElems; i < nWords * int3_kLanesPerWord; ++i)
		int3_set(dst, i, 0);
}

static inline bool int3_anyZeroDivisor(const uint64_t* src, size_t nElems) {
	for (size_t i = 0; i < nElems; ++i)
		if (int3_get(src, i) == 0)
			return true;
	return false;
}

// ---- portable scalar path --------------------------------------------------

static inline void int3_tableApplyScalar(uint64_t* dst, const uint64_t* src, size_t nElems,
                                         const uint8_t table[64]) {
	for (size_t i = 0; i < nElems; ++i) {
		uint8_t a = int3_get(dst, i);
		uint8_t b = int3_get(src, i);
		int3_set(dst, i, table[(size_t)a | ((size_t)b << 3)]);
	}
	int3_clearPadding(dst, nElems);
}

static inline void int3_addScalarPath(uint64_t* dst, const uint64_t* src, size_t nWords) {
	for (size_t w = 0; w < nWords; ++w)
		dst[w] = int3_addWord(dst[w] & int3_kValueMask, src[w] & int3_kValueMask);
}

static inline void int3_subScalarPath(uint64_t* dst, const uint64_t* src, size_t nWords) {
	for (size_t w = 0; w < nWords; ++w)
		dst[w] = int3_subWord(dst[w] & int3_kValueMask, src[w] & int3_kValueMask);
}

// ---- AVX-512 (same tables) -------------------------------------------------

#if LIBEXCESSIVE_INT3_AVX512

static inline __m512i int3_highNibblesAsBytes(__m512i v) {
	return _mm512_and_si512(_mm512_srli_epi16(v, 4), _mm512_set1_epi8(0x0F));
}

static inline __m512i int3_valuesLo3(__m512i v) {
	return _mm512_and_si512(v, _mm512_set1_epi8(0x07));
}

static inline __m512i int3_valuesHi3(__m512i v) {
	return _mm512_and_si512(int3_highNibblesAsBytes(v), _mm512_set1_epi8(0x07));
}

static inline __m512i int3_packLoHi(__m512i lo, __m512i hi) {
	__m512i hiShift = _mm512_and_si512(_mm512_slli_epi16(hi, 4), _mm512_set1_epi8((char)0xF0));
	return _mm512_or_si512(lo, hiShift);
}

static inline void int3_tableApplyAvx512(uint64_t* dst, const uint64_t* src, size_t nElems,
                                         const uint8_t table[64]) {
	const size_t nWords = int3_wordCount(nElems);
	const size_t nBytes = nWords * sizeof(uint64_t);
	uint8_t* db = reinterpret_cast<uint8_t*>(dst);
	const uint8_t* sb = reinterpret_cast<const uint8_t*>(src);
	const __m512i tab = _mm512_load_si512(reinterpret_cast<const void*>(table));
	const __m512i idxMask = _mm512_set1_epi8(0x3F);

	size_t off = 0;
	for (; off + 64 <= nBytes; off += 64) {
		__m512i va = _mm512_loadu_si512(db + off);
		__m512i vb = _mm512_loadu_si512(sb + off);

		__m512i a_lo = int3_valuesLo3(va);
		__m512i b_lo = int3_valuesLo3(vb);
		__m512i a_hi = int3_valuesHi3(va);
		__m512i b_hi = int3_valuesHi3(vb);

		__m512i idx_lo = _mm512_and_si512(
			_mm512_or_si512(a_lo, _mm512_slli_epi16(b_lo, 3)), idxMask);
		__m512i idx_hi = _mm512_and_si512(
			_mm512_or_si512(a_hi, _mm512_slli_epi16(b_hi, 3)), idxMask);

		__m512i p_lo = _mm512_permutexvar_epi8(idx_lo, tab);
		__m512i p_hi = _mm512_permutexvar_epi8(idx_hi, tab);
		_mm512_storeu_si512(db + off, int3_packLoHi(p_lo, p_hi));
	}

	const size_t doneElems = off * 2;
	for (size_t i = doneElems; i < nElems; ++i) {
		uint8_t a = int3_get(dst, i);
		uint8_t b = int3_get(src, i);
		int3_set(dst, i, table[(size_t)a | ((size_t)b << 3)]);
	}
	int3_clearPadding(dst, nElems);
}

static inline void int3_addAvx512(uint64_t* dst, const uint64_t* src, size_t nWords) {
	const __m512i mask = _mm512_set1_epi64((long long)int3_kValueMask);
	size_t w = 0;
	for (; w + 8 <= nWords; w += 8) {
		__m512i a = _mm512_and_si512(_mm512_loadu_si512(dst + w), mask);
		__m512i b = _mm512_and_si512(_mm512_loadu_si512(src + w), mask);
		_mm512_storeu_si512(dst + w, _mm512_and_si512(_mm512_add_epi64(a, b), mask));
	}
	for (; w < nWords; ++w)
		dst[w] = int3_addWord(dst[w] & int3_kValueMask, src[w] & int3_kValueMask);
}

static inline void int3_subAvx512(uint64_t* dst, const uint64_t* src, size_t nWords) {
	const __m512i mask = _mm512_set1_epi64((long long)int3_kValueMask);
	const __m512i pad = _mm512_set1_epi64((long long)int3_kPadMask);
	size_t w = 0;
	for (; w + 8 <= nWords; w += 8) {
		__m512i a = _mm512_and_si512(_mm512_loadu_si512(dst + w), mask);
		__m512i b = _mm512_and_si512(_mm512_loadu_si512(src + w), mask);
		__m512i t = _mm512_sub_epi64(_mm512_or_si512(a, pad), b);
		_mm512_storeu_si512(dst + w, _mm512_and_si512(t, mask));
	}
	for (; w < nWords; ++w)
		dst[w] = int3_subWord(dst[w] & int3_kValueMask, src[w] & int3_kValueMask);
}

#endif // LIBEXCESSIVE_INT3_AVX512

// ---- bulk kernels (called from NDArray.cpp) --------------------------------

static inline void int3_add(uint64_t* dst, const uint64_t* src, size_t nElems) {
	const size_t nWords = int3_wordCount(nElems);
#if LIBEXCESSIVE_INT3_AVX512
	if (nWords >= 8)
		int3_addAvx512(dst, src, nWords);
	else
		int3_addScalarPath(dst, src, nWords);
#else
	int3_addScalarPath(dst, src, nWords);
#endif
	int3_clearPadding(dst, nElems);
}

static inline void int3_sub(uint64_t* dst, const uint64_t* src, size_t nElems) {
	const size_t nWords = int3_wordCount(nElems);
#if LIBEXCESSIVE_INT3_AVX512
	if (nWords >= 8)
		int3_subAvx512(dst, src, nWords);
	else
		int3_subScalarPath(dst, src, nWords);
#else
	int3_subScalarPath(dst, src, nWords);
#endif
	int3_clearPadding(dst, nElems);
}

static inline void int3_mul(uint64_t* dst, const uint64_t* src, size_t nElems) {
#if LIBEXCESSIVE_INT3_AVX512
	if (int3_wordCount(nElems) * sizeof(uint64_t) >= 64)
		int3_tableApplyAvx512(dst, src, nElems, int3_kMulTable);
	else
		int3_tableApplyScalar(dst, src, nElems, int3_kMulTable);
#else
	int3_tableApplyScalar(dst, src, nElems, int3_kMulTable);
#endif
}

static inline void int3_div(uint64_t* dst, const uint64_t* src, size_t nElems) {
	if (int3_anyZeroDivisor(src, nElems))
		throw std::invalid_argument("NDArray: division by zero");
#if LIBEXCESSIVE_INT3_AVX512
	if (int3_wordCount(nElems) * sizeof(uint64_t) >= 64)
		int3_tableApplyAvx512(dst, src, nElems, int3_kDivTable);
	else
		int3_tableApplyScalar(dst, src, nElems, int3_kDivTable);
#else
	int3_tableApplyScalar(dst, src, nElems, int3_kDivTable);
#endif
}

static inline void int3_addScalar(uint64_t* dst, uint8_t pattern3, size_t nElems) {
	const uint64_t s = int3_splatWord(pattern3);
	const size_t nWords = int3_wordCount(nElems);
	for (size_t w = 0; w < nWords; ++w)
		dst[w] = int3_addWord(dst[w] & int3_kValueMask, s);
	int3_clearPadding(dst, nElems);
}

static inline void int3_subScalar(uint64_t* dst, uint8_t pattern3, size_t nElems) {
	const uint64_t s = int3_splatWord(pattern3);
	const size_t nWords = int3_wordCount(nElems);
	for (size_t w = 0; w < nWords; ++w)
		dst[w] = int3_subWord(dst[w] & int3_kValueMask, s);
	int3_clearPadding(dst, nElems);
}

static inline void int3_mulScalar(uint64_t* dst, uint8_t pattern3, size_t nElems) {
	pattern3 &= 7;
	for (size_t i = 0; i < nElems; ++i)
		int3_set(dst, i, int3_kMulTable[(size_t)int3_get(dst, i) | ((size_t)pattern3 << 3)]);
	int3_clearPadding(dst, nElems);
}

static inline void int3_divScalar(uint64_t* dst, uint8_t pattern3, size_t nElems) {
	pattern3 &= 7;
	if (pattern3 == 0)
		throw std::invalid_argument("NDArray: division by zero");
	for (size_t i = 0; i < nElems; ++i)
		int3_set(dst, i, int3_kDivTable[(size_t)int3_get(dst, i) | ((size_t)pattern3 << 3)]);
	int3_clearPadding(dst, nElems);
}

#endif // LIBEXCESSIVE_SRC_NDARRAY_INT3_H
