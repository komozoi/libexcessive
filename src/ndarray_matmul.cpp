// Tiled matmul / gemv. Last two axes are the matrix (NumPy); leading axis is batch.

#include "ndarray_matmul.h"

#include "NDArray.h"
#include "fs/FdHandle.h"
#include "ndarray_int3.h"

#include <cstring>
#include <cstdlib>
#include <stdexcept>

#if defined(_MSC_VER)
#define NDM_RESTRICT __restrict
#else
#define NDM_RESTRICT __restrict__
#endif

#if defined(__AVX2__)
#define NDM_AVX2 1
#include <immintrin.h>
#else
#define NDM_AVX2 0
#endif

#if defined(__FMA__)
#define NDM_FMA 1
#else
#define NDM_FMA 0
#endif

#if defined(__AVX512VPOPCNTDQ__)
#define NDM_VPOPCNT 1
#include <immintrin.h>
#else
#define NDM_VPOPCNT 0
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define NDM_NEON 1
#include <arm_neon.h>
#else
#define NDM_NEON 0
#endif


static const int kMC = 64;
static const int kKC = 256;
static const int kNR = 8;
static const int kMR = 8;


static NDArrayType matmulResultType(NDArrayType t) {
	switch (t) {
		case BINARY:
		case INT3:
		case INT8:
		case UINT8:
			return INT32;
		case INT32:
			return INT64;
		case INT64:
		case F32:
		case F64:
		case UINT256:
			return t;
		default:
			throw std::invalid_argument("NDArray::matmul - unsupported type");
	}
}

static size_t denseElemSize(NDArrayType t) {
	switch (t) {
		case F32:
		case INT32:
			return 4;
		case F64:
		case INT64:
			return 8;
		case INT8:
		case UINT8:
			return 1;
		case UINT256:
			return 32;
		default:
			return 0;
	}
}

static void* alloc64(size_t bytes) {
	if (bytes == 0)
		bytes = 64;
	void* p = nullptr;
#if defined(_MSC_VER)
	p = _aligned_malloc(bytes, 64);
	if (!p)
		throw std::bad_alloc();
#else
	if (posix_memalign(&p, 64, bytes) != 0)
		throw std::bad_alloc();
#endif
	return p;
}

static void free64(void* p) {
#if defined(_MSC_VER)
	_aligned_free(p);
#else
	free(p);
#endif
}

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

static bool viewIsWrap(const NDArrayView& v) {
	if (!v.sharedBuffer() || !v.sharedBuffer().get())
		return false;
	return !v.sharedBuffer().get()->ownsData;
}

// Advise [row0, row0+rows) × [col0, col0+cols) of a row-major rowsTot×colsTot
// matrix. Length is ((rows-1)*colsTot + cols) elements from the first lane —
// never past the allocation.
static void advisePanel(const void* base, size_t elemBytes, int colsTot,
                        int row0, int rows, int col0, int cols, bool wrap) {
	if (!wrap || !base || rows <= 0 || cols <= 0 || elemBytes == 0)
		return;
	const char* p = (const char*)base + ((size_t)row0 * (size_t)colsTot + (size_t)col0) * elemBytes;
	size_t bytes = ((size_t)(rows - 1) * (size_t)colsTot + (size_t)cols) * elemBytes;
	memoryAdviseWillNeed((void*)p, bytes);
}


// ---- F32 8×8 microkernel ---------------------------------------------------

#if NDM_AVX2
static inline __m256 ndmFmadd(__m256 a, __m256 b, __m256 c) {
#if NDM_FMA
	return _mm256_fmadd_ps(a, b, c);
#else
	return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}

static void kernelF32_8x8(const float* NDM_RESTRICT A, const float* NDM_RESTRICT B,
                          float* NDM_RESTRICT C, int ldc, int kb) {
	__m256 c0 = _mm256_loadu_ps(C + 0 * ldc);
	__m256 c1 = _mm256_loadu_ps(C + 1 * ldc);
	__m256 c2 = _mm256_loadu_ps(C + 2 * ldc);
	__m256 c3 = _mm256_loadu_ps(C + 3 * ldc);
	__m256 c4 = _mm256_loadu_ps(C + 4 * ldc);
	__m256 c5 = _mm256_loadu_ps(C + 5 * ldc);
	__m256 c6 = _mm256_loadu_ps(C + 6 * ldc);
	__m256 c7 = _mm256_loadu_ps(C + 7 * ldc);
	for (int k = 0; k < kb; ++k) {
		__m256 bv = _mm256_loadu_ps(B + (size_t)k * (size_t)kNR);
		c0 = ndmFmadd(_mm256_broadcast_ss(A + 0 * (size_t)kKC + (size_t)k), bv, c0);
		c1 = ndmFmadd(_mm256_broadcast_ss(A + 1 * (size_t)kKC + (size_t)k), bv, c1);
		c2 = ndmFmadd(_mm256_broadcast_ss(A + 2 * (size_t)kKC + (size_t)k), bv, c2);
		c3 = ndmFmadd(_mm256_broadcast_ss(A + 3 * (size_t)kKC + (size_t)k), bv, c3);
		c4 = ndmFmadd(_mm256_broadcast_ss(A + 4 * (size_t)kKC + (size_t)k), bv, c4);
		c5 = ndmFmadd(_mm256_broadcast_ss(A + 5 * (size_t)kKC + (size_t)k), bv, c5);
		c6 = ndmFmadd(_mm256_broadcast_ss(A + 6 * (size_t)kKC + (size_t)k), bv, c6);
		c7 = ndmFmadd(_mm256_broadcast_ss(A + 7 * (size_t)kKC + (size_t)k), bv, c7);
	}
	_mm256_storeu_ps(C + 0 * ldc, c0);
	_mm256_storeu_ps(C + 1 * ldc, c1);
	_mm256_storeu_ps(C + 2 * ldc, c2);
	_mm256_storeu_ps(C + 3 * ldc, c3);
	_mm256_storeu_ps(C + 4 * ldc, c4);
	_mm256_storeu_ps(C + 5 * ldc, c5);
	_mm256_storeu_ps(C + 6 * ldc, c6);
	_mm256_storeu_ps(C + 7 * ldc, c7);
}
#endif

#if NDM_NEON
static void kernelF32_8x8_neon(const float* NDM_RESTRICT A, const float* NDM_RESTRICT B,
                               float* NDM_RESTRICT C, int ldc, int kb) {
	float32x4_t c00 = vld1q_f32(C + 0 * ldc);
	float32x4_t c01 = vld1q_f32(C + 0 * ldc + 4);
	float32x4_t c10 = vld1q_f32(C + 1 * ldc);
	float32x4_t c11 = vld1q_f32(C + 1 * ldc + 4);
	float32x4_t c20 = vld1q_f32(C + 2 * ldc);
	float32x4_t c21 = vld1q_f32(C + 2 * ldc + 4);
	float32x4_t c30 = vld1q_f32(C + 3 * ldc);
	float32x4_t c31 = vld1q_f32(C + 3 * ldc + 4);
	float32x4_t c40 = vld1q_f32(C + 4 * ldc);
	float32x4_t c41 = vld1q_f32(C + 4 * ldc + 4);
	float32x4_t c50 = vld1q_f32(C + 5 * ldc);
	float32x4_t c51 = vld1q_f32(C + 5 * ldc + 4);
	float32x4_t c60 = vld1q_f32(C + 6 * ldc);
	float32x4_t c61 = vld1q_f32(C + 6 * ldc + 4);
	float32x4_t c70 = vld1q_f32(C + 7 * ldc);
	float32x4_t c71 = vld1q_f32(C + 7 * ldc + 4);
	for (int k = 0; k < kb; ++k) {
		const float* bp = B + (size_t)k * (size_t)kNR;
		float32x4_t b0 = vld1q_f32(bp);
		float32x4_t b1 = vld1q_f32(bp + 4);
		float32x4_t a;
		a = vdupq_n_f32(A[0 * kKC + k]); c00 = vfmaq_f32(c00, a, b0); c01 = vfmaq_f32(c01, a, b1);
		a = vdupq_n_f32(A[1 * kKC + k]); c10 = vfmaq_f32(c10, a, b0); c11 = vfmaq_f32(c11, a, b1);
		a = vdupq_n_f32(A[2 * kKC + k]); c20 = vfmaq_f32(c20, a, b0); c21 = vfmaq_f32(c21, a, b1);
		a = vdupq_n_f32(A[3 * kKC + k]); c30 = vfmaq_f32(c30, a, b0); c31 = vfmaq_f32(c31, a, b1);
		a = vdupq_n_f32(A[4 * kKC + k]); c40 = vfmaq_f32(c40, a, b0); c41 = vfmaq_f32(c41, a, b1);
		a = vdupq_n_f32(A[5 * kKC + k]); c50 = vfmaq_f32(c50, a, b0); c51 = vfmaq_f32(c51, a, b1);
		a = vdupq_n_f32(A[6 * kKC + k]); c60 = vfmaq_f32(c60, a, b0); c61 = vfmaq_f32(c61, a, b1);
		a = vdupq_n_f32(A[7 * kKC + k]); c70 = vfmaq_f32(c70, a, b0); c71 = vfmaq_f32(c71, a, b1);
	}
	vst1q_f32(C + 0 * ldc, c00); vst1q_f32(C + 0 * ldc + 4, c01);
	vst1q_f32(C + 1 * ldc, c10); vst1q_f32(C + 1 * ldc + 4, c11);
	vst1q_f32(C + 2 * ldc, c20); vst1q_f32(C + 2 * ldc + 4, c21);
	vst1q_f32(C + 3 * ldc, c30); vst1q_f32(C + 3 * ldc + 4, c31);
	vst1q_f32(C + 4 * ldc, c40); vst1q_f32(C + 4 * ldc + 4, c41);
	vst1q_f32(C + 5 * ldc, c50); vst1q_f32(C + 5 * ldc + 4, c51);
	vst1q_f32(C + 6 * ldc, c60); vst1q_f32(C + 6 * ldc + 4, c61);
	vst1q_f32(C + 7 * ldc, c70); vst1q_f32(C + 7 * ldc + 4, c71);
}
#endif


// Packed A is [kMC][kKC], packed B is [kKC][kNR] (always NR-wide, padded).
template <typename Lane, typename Acc, typename Out>
static void gemmTiled(const Lane* NDM_RESTRICT A, const Lane* NDM_RESTRICT B, Out* NDM_RESTRICT C,
                      int M, int N, int K, bool wrapA, bool wrapB) {
	memset(C, 0, (size_t)M * (size_t)N * sizeof(Out));
	Lane* Bp = (Lane*)alloc64((size_t)kKC * (size_t)kNR * sizeof(Lane));
	Lane* Ap = (Lane*)alloc64((size_t)kMC * (size_t)kKC * sizeof(Lane));
	memset(Bp, 0, (size_t)kKC * (size_t)kNR * sizeof(Lane));

	for (int kc = 0; kc < K; kc += kKC) {
		int kb = K - kc;
		if (kb > kKC)
			kb = kKC;
		advisePanel(B, sizeof(Lane), N, kc, kb, 0, N, wrapB);

		for (int jc = 0; jc < N; jc += kNR) {
			int nb = N - jc;
			if (nb > kNR)
				nb = kNR;
			for (int k = 0; k < kb; ++k) {
				memcpy(Bp + (size_t)k * (size_t)kNR,
				       B + (size_t)(kc + k) * (size_t)N + (size_t)jc,
				       (size_t)nb * sizeof(Lane));
				if (nb < kNR)
					memset(Bp + (size_t)k * (size_t)kNR + (size_t)nb, 0,
					       (size_t)(kNR - nb) * sizeof(Lane));
			}

			for (int ic = 0; ic < M; ic += kMC) {
				int mb = M - ic;
				if (mb > kMC)
					mb = kMC;
				advisePanel(A, sizeof(Lane), K, ic, mb, kc, kb, wrapA);
				for (int i = 0; i < mb; ++i)
					memcpy(Ap + (size_t)i * (size_t)kKC,
					       A + (size_t)(ic + i) * (size_t)K + (size_t)kc,
					       (size_t)kb * sizeof(Lane));

				for (int ir = 0; ir < mb; ) {
					int mr = mb - ir;
					if (mr > kMR)
						mr = kMR;
#if NDM_AVX2
					if (sizeof(Lane) == 4 && sizeof(Out) == 4 && mr == 8 && nb == 8) {
						kernelF32_8x8((const float*)(Ap + (size_t)ir * (size_t)kKC),
						              (const float*)Bp,
						              (float*)(C + (size_t)(ic + ir) * (size_t)N + (size_t)jc),
						              N, kb);
						ir += 8;
						continue;
					}
#endif
#if NDM_NEON
					if (sizeof(Lane) == 4 && sizeof(Out) == 4 && mr == 8 && nb == 8) {
						kernelF32_8x8_neon((const float*)(Ap + (size_t)ir * (size_t)kKC),
						                   (const float*)Bp,
						                   (float*)(C + (size_t)(ic + ir) * (size_t)N + (size_t)jc),
						                   N, kb);
						ir += 8;
						continue;
					}
#endif
					for (int i = 0; i < mr; ++i) {
						Out* crow = C + (size_t)(ic + ir + i) * (size_t)N + (size_t)jc;
						const Lane* arow = Ap + (size_t)(ir + i) * (size_t)kKC;
						for (int j = 0; j < nb; ++j) {
							Acc acc = (Acc)crow[j];
							for (int k = 0; k < kb; ++k)
								acc += (Acc)arow[k] * (Acc)Bp[(size_t)k * (size_t)kNR + (size_t)j];
							crow[j] = (Out)acc;
						}
					}
					ir += mr;
				}
			}
		}
	}
	free64(Ap);
	free64(Bp);
}

#if defined(__SIZEOF_INT128__)
template <>
void gemmTiled<int64_t, int64_t, int64_t>(const int64_t* NDM_RESTRICT A, const int64_t* NDM_RESTRICT B,
                                          int64_t* NDM_RESTRICT C, int M, int N, int K,
                                          bool wrapA, bool wrapB) {
	memset(C, 0, (size_t)M * (size_t)N * sizeof(int64_t));
	int64_t* Bp = (int64_t*)alloc64((size_t)kKC * (size_t)kNR * sizeof(int64_t));
	int64_t* Ap = (int64_t*)alloc64((size_t)kMC * (size_t)kKC * sizeof(int64_t));
	memset(Bp, 0, (size_t)kKC * (size_t)kNR * sizeof(int64_t));
	for (int kc = 0; kc < K; kc += kKC) {
		int kb = K - kc;
		if (kb > kKC)
			kb = kKC;
		advisePanel(B, sizeof(int64_t), N, kc, kb, 0, N, wrapB);
		for (int jc = 0; jc < N; jc += kNR) {
			int nb = N - jc;
			if (nb > kNR)
				nb = kNR;
			for (int k = 0; k < kb; ++k) {
				memcpy(Bp + (size_t)k * (size_t)kNR,
				       B + (size_t)(kc + k) * (size_t)N + (size_t)jc,
				       (size_t)nb * sizeof(int64_t));
				if (nb < kNR)
					memset(Bp + (size_t)k * (size_t)kNR + (size_t)nb, 0,
					       (size_t)(kNR - nb) * sizeof(int64_t));
			}
			for (int ic = 0; ic < M; ic += kMC) {
				int mb = M - ic;
				if (mb > kMC)
					mb = kMC;
				advisePanel(A, sizeof(int64_t), K, ic, mb, kc, kb, wrapA);
				for (int i = 0; i < mb; ++i)
					memcpy(Ap + (size_t)i * (size_t)kKC,
					       A + (size_t)(ic + i) * (size_t)K + (size_t)kc,
					       (size_t)kb * sizeof(int64_t));
				for (int ir = 0; ir < mb; ++ir) {
					int64_t* crow = C + (size_t)(ic + ir) * (size_t)N + (size_t)jc;
					const int64_t* arow = Ap + (size_t)ir * (size_t)kKC;
					for (int j = 0; j < nb; ++j) {
						__int128 acc = (__int128)crow[j];
						for (int k = 0; k < kb; ++k)
							acc += (__int128)arow[k] * (__int128)Bp[(size_t)k * (size_t)kNR + (size_t)j];
						crow[j] = (int64_t)acc;
					}
				}
			}
		}
	}
	free64(Ap);
	free64(Bp);
}
#endif


// ---- gemv (N == 1): no B pack ---------------------------------------------

template <typename Lane, typename Acc, typename Out>
static void gemvDense(const Lane* A, const Lane* x, Out* y, int M, int K) {
	for (int i = 0; i < M; ++i) {
		Acc acc = 0;
		const Lane* arow = A + (size_t)i * (size_t)K;
#if NDM_AVX2
		if (sizeof(Lane) == 4 && sizeof(Out) == 4) {
			__m256 vacc = _mm256_setzero_ps();
			int k = 0;
			for (; k + 8 <= K; k += 8) {
				__m256 av = _mm256_loadu_ps((const float*)(arow + k));
				__m256 xv = _mm256_loadu_ps((const float*)(x + k));
				vacc = ndmFmadd(av, xv, vacc);
			}
			alignas(32) float tmp[8];
			_mm256_store_ps(tmp, vacc);
			float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
			for (; k < K; ++k)
				s += (float)arow[k] * (float)x[k];
			y[i] = (Out)s;
			continue;
		}
#endif
		for (int k = 0; k < K; ++k)
			acc += (Acc)arow[k] * (Acc)x[k];
		y[i] = (Out)acc;
	}
}


// ---- INT3 / BINARY pack ----------------------------------------------------

static void unpackInt3(int8_t* dst, const uint64_t* src, size_t n) {
	size_t i = 0;
	const size_t words = n / 16;
	for (size_t w = 0; w < words; ++w) {
		uint64_t x = src[w];
		for (int lane = 0; lane < 16; ++lane) {
			uint8_t p = (uint8_t)((x >> (unsigned)(lane * 4)) & 7u);
			dst[i++] = (int8_t)(p >= 4 ? (int)p - 8 : (int)p);
		}
	}
	for (; i < n; ++i)
		dst[i] = (int8_t)int3_getSigned(src, i);
}

static void packBitRow(uint64_t* dst, const uint64_t* src, size_t bit0, int K) {
	const int words = (K + 63) / 64;
	memset(dst, 0, (size_t)words * sizeof(uint64_t));
	if ((bit0 & 63u) == 0) {
		size_t w0 = bit0 >> 6;
		int full = K / 64;
		if (full)
			memcpy(dst, src + w0, (size_t)full * sizeof(uint64_t));
		if (K & 63)
			dst[full] = src[w0 + (size_t)full] & (((uint64_t)1 << (K & 63)) - 1u);
		return;
	}
	for (int k = 0; k < K; ++k) {
		size_t srcBit = bit0 + (size_t)k;
		if ((src[srcBit >> 6] >> (srcBit & 63u)) & 1u)
			dst[k >> 6] |= (uint64_t)1 << (k & 63);
	}
}

static void gemmInt3(const uint64_t* Aw, const uint64_t* Bw, int32_t* C, int M, int N, int K,
                     bool wrap) {
	int8_t* A = (int8_t*)alloc64((size_t)M * (size_t)K);
	int8_t* B = (int8_t*)alloc64((size_t)K * (size_t)N);
	if (wrap) {
		memoryAdviseWillNeed((void*)Aw, int3_bufferBytes((size_t)M * (size_t)K));
		memoryAdviseWillNeed((void*)Bw, int3_bufferBytes((size_t)K * (size_t)N));
	}
	unpackInt3(A, Aw, (size_t)M * (size_t)K);
	unpackInt3(B, Bw, (size_t)K * (size_t)N);
	int64_t* acc = (int64_t*)alloc64((size_t)M * (size_t)N * sizeof(int64_t));
	gemmTiled<int8_t, int64_t, int64_t>(A, B, acc, M, N, K, false, false);
	for (int i = 0; i < M * N; ++i)
		C[i] = (int32_t)acc[i];
	free64(acc);
	free64(A);
	free64(B);
}

static void gemmBinary(const uint64_t* Aw, const uint64_t* Bw, int32_t* C, int M, int N, int K,
                       bool wrap) {
	memset(C, 0, (size_t)M * (size_t)N * sizeof(int32_t));
	if (wrap) {
		memoryAdviseWillNeed((void*)Aw, ((size_t)M * (size_t)K + 63) / 64 * 8);
		memoryAdviseWillNeed((void*)Bw, ((size_t)K * (size_t)N + 63) / 64 * 8);
	}

	int kWords = (K + 63) / 64;
	uint64_t* aRows = (uint64_t*)calloc((size_t)M * (size_t)kWords, sizeof(uint64_t));
	uint64_t* bCols = (uint64_t*)calloc((size_t)N * (size_t)kWords, sizeof(uint64_t));
	if (!aRows || !bCols) {
		free(aRows);
		free(bCols);
		throw std::bad_alloc();
	}
	for (int i = 0; i < M; ++i)
		packBitRow(aRows + (size_t)i * (size_t)kWords, Aw, (size_t)i * (size_t)K, K);
	for (int j = 0; j < N; ++j) {
		uint64_t* col = bCols + (size_t)j * (size_t)kWords;
		for (int k = 0; k < K; ++k) {
			size_t bit = (size_t)k * (size_t)N + (size_t)j;
			if ((Bw[bit >> 6] >> (bit & 63u)) & 1u)
				col[k >> 6] |= (uint64_t)1 << (k & 63);
		}
	}

	for (int i = 0; i < M; ++i) {
		const uint64_t* ar = aRows + (size_t)i * (size_t)kWords;
		for (int j = 0; j < N; ++j) {
			const uint64_t* br = bCols + (size_t)j * (size_t)kWords;
			size_t acc = 0;
			int w = 0;
#if NDM_VPOPCNT
			for (; w + 8 <= kWords; w += 8) {
				__m512i va = _mm512_loadu_si512((const void*)(ar + w));
				__m512i vb = _mm512_loadu_si512((const void*)(br + w));
				__m512i p = _mm512_popcnt_epi64(_mm512_and_si512(va, vb));
				acc += (size_t)_mm512_reduce_add_epi64(p);
			}
#endif
			for (; w < kWords; ++w)
				acc += popcnt64(ar[w] & br[w]);
			C[(size_t)i * (size_t)N + (size_t)j] = (int32_t)acc;
		}
	}
	free(aRows);
	free(bCols);
}

static void gemmU256(const uint256_t* A, const uint256_t* B, uint256_t* C, int M, int N, int K) {
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			uint256_t acc(0);
			for (int k = 0; k < K; ++k)
				acc = acc + A[(size_t)i * (size_t)K + (size_t)k] * B[(size_t)k * (size_t)N + (size_t)j];
			C[(size_t)i * (size_t)N + (size_t)j] = acc;
		}
	}
}


// ---- materialize a dense (M,K) or (K,N) plane ------------------------------

static bool isContig2d(const NDArrayView& v, int rows, int cols) {
	if (!v.isContiguous() || v.getShape().size() != 2)
		return false;
	return v.getShape().get(0) == rows && v.getShape().get(1) == cols;
}

static NDArray dense2d(const NDArrayView& v, int rows, int cols, int batch, int b) {
	if (batch <= 1 && isContig2d(v, rows, cols))
		return v.copy();

	if (batch > 1 && v.isContiguous() && v.getShape().size() == 3 &&
	    v.getShape().get(0) == batch && v.getShape().get(1) == rows &&
	    v.getShape().get(2) == cols) {
		const size_t es = denseElemSize(v.getType());
		if (es != 0 && v.data()) {
			NDArray out({rows, cols}, v.getType());
			const char* src = (const char*)v.data() +
			                  ((size_t)v.getOffset() + (size_t)b * (size_t)rows * (size_t)cols) * es;
			memcpy((void*)out.data(), src, (size_t)rows * (size_t)cols * es);
			return out;
		}
	}

	NDArray out({rows, cols}, v.getType());
	const ArrayList<size_t>& st = v.getStrides();
	const int r = v.getShape().size();
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			size_t e = v.getOffset();
			if (r == 1)
				e += (size_t)(rows == 1 ? j : i) * (st.size() > 0 ? st.get(0) : 1);
			else if (r == 2)
				e += (size_t)i * st.get(0) + (size_t)j * st.get(1);
			else {
				e += (size_t)b * st.get(0) + (size_t)i * st.get(1) + (size_t)j * st.get(2);
			}
			switch (v.getType()) {
				case F32:
					out.set({i, j}, ((const float*)v.data())[e]);
					break;
				case F64:
					out.set({i, j}, ((const double*)v.data())[e]);
					break;
				case UINT256:
					out.set({i, j}, ((const uint256_t*)v.data())[e]);
					break;
				case INT64:
					out.set({i, j}, ((const int64_t*)v.data())[e]);
					break;
				case INT32:
					out.set({i, j}, ((const int32_t*)v.data())[e]);
					break;
				case INT8:
					out.set({i, j}, (int)((const int8_t*)v.data())[e]);
					break;
				case UINT8:
					out.set({i, j}, (int)((const uint8_t*)v.data())[e]);
					break;
				case INT3:
					out.set({i, j}, int3_getSigned((const uint64_t*)v.data(), e));
					break;
				case BINARY:
					out.set({i, j}, (int)((((const uint64_t*)v.data())[e >> 6] >> (e & 63u)) & 1u));
					break;
				default:
					break;
			}
		}
	}
	return out;
}

static void storePlane(NDArray& c, const NDArray& plane, int batch, int b) {
	int M = plane.shape.get(0);
	int N = plane.shape.get(1);
	if (batch > 1 && c.isContiguous() && c.shape.size() == 3) {
		const size_t es = denseElemSize(c.type);
		if (es != 0 && c.data()) {
			char* dst = (char*)c.data() + (size_t)b * (size_t)M * (size_t)N * es;
			memcpy(dst, plane.data(), (size_t)M * (size_t)N * es);
			return;
		}
	}
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			switch (c.type) {
				case F32:
					if (batch > 1)
						c.set({b, i, j}, plane.get<float>({i, j}));
					else
						c.set({i, j}, plane.get<float>({i, j}));
					break;
				case F64:
					if (batch > 1)
						c.set({b, i, j}, plane.get<double>({i, j}));
					else
						c.set({i, j}, plane.get<double>({i, j}));
					break;
				case UINT256:
					if (batch > 1)
						c.set({b, i, j}, plane.get<uint256_t>({i, j}));
					else
						c.set({i, j}, plane.get<uint256_t>({i, j}));
					break;
				case INT64:
					if (batch > 1)
						c.set({b, i, j}, plane.get<int64_t>({i, j}));
					else
						c.set({i, j}, plane.get<int64_t>({i, j}));
					break;
				default:
					if (batch > 1)
						c.set({b, i, j}, plane.get<int>({i, j}));
					else
						c.set({i, j}, plane.get<int>({i, j}));
					break;
			}
		}
	}
}

static NDArray gemmOwned(const NDArray& a, const NDArray& b, bool wrapA, bool wrapB) {
	int M = a.shape.get(0);
	int K = a.shape.get(1);
	int N = b.shape.get(1);
	NDArray c({M, N}, matmulResultType(a.type));
	const void* ap = a.data();
	const void* bp = b.data();
	void* cp = c.data();
	if (M == 0 || N == 0)
		return c;
	if (!ap || !bp)
		throw std::invalid_argument("NDArray::matmul - null buffer");

	if (N == 1) {
		switch (a.type) {
			case F32:
				gemvDense<float, float, float>((const float*)ap, (const float*)bp, (float*)cp, M, K);
				return c;
			case F64:
				gemvDense<double, double, double>((const double*)ap, (const double*)bp, (double*)cp, M, K);
				return c;
			case INT8:
				gemvDense<int8_t, int64_t, int32_t>((const int8_t*)ap, (const int8_t*)bp, (int32_t*)cp, M, K);
				return c;
			case UINT8:
				gemvDense<uint8_t, int64_t, int32_t>((const uint8_t*)ap, (const uint8_t*)bp, (int32_t*)cp, M, K);
				return c;
			case INT32:
				gemvDense<int32_t, int64_t, int64_t>((const int32_t*)ap, (const int32_t*)bp, (int64_t*)cp, M, K);
				return c;
			case INT64:
				gemvDense<int64_t, int64_t, int64_t>((const int64_t*)ap, (const int64_t*)bp, (int64_t*)cp, M, K);
				return c;
			default:
				break;
		}
	}

	switch (a.type) {
		case F32:
			gemmTiled<float, float, float>((const float*)ap, (const float*)bp, (float*)cp, M, N, K, wrapA, wrapB);
			break;
		case F64:
			gemmTiled<double, double, double>((const double*)ap, (const double*)bp, (double*)cp, M, N, K, wrapA, wrapB);
			break;
		case INT8: {
			int64_t* acc = (int64_t*)alloc64((size_t)M * (size_t)N * sizeof(int64_t));
			gemmTiled<int8_t, int64_t, int64_t>((const int8_t*)ap, (const int8_t*)bp, acc, M, N, K, wrapA, wrapB);
			for (int i = 0; i < M * N; ++i)
				((int32_t*)cp)[i] = (int32_t)acc[i];
			free64(acc);
			break;
		}
		case UINT8: {
			int64_t* acc = (int64_t*)alloc64((size_t)M * (size_t)N * sizeof(int64_t));
			gemmTiled<uint8_t, int64_t, int64_t>((const uint8_t*)ap, (const uint8_t*)bp, acc, M, N, K, wrapA, wrapB);
			for (int i = 0; i < M * N; ++i)
				((int32_t*)cp)[i] = (int32_t)acc[i];
			free64(acc);
			break;
		}
		case INT32:
			gemmTiled<int32_t, int64_t, int64_t>((const int32_t*)ap, (const int32_t*)bp, (int64_t*)cp,
			                                     M, N, K, wrapA, wrapB);
			break;
		case INT64:
			gemmTiled<int64_t, int64_t, int64_t>((const int64_t*)ap, (const int64_t*)bp, (int64_t*)cp,
			                                     M, N, K, wrapA, wrapB);
			break;
		case UINT256:
			gemmU256((const uint256_t*)ap, (const uint256_t*)bp, (uint256_t*)cp, M, N, K);
			break;
		case INT3:
			gemmInt3((const uint64_t*)ap, (const uint64_t*)bp, (int32_t*)cp, M, N, K, wrapA || wrapB);
			break;
		case BINARY:
			gemmBinary((const uint64_t*)ap, (const uint64_t*)bp, (int32_t*)cp, M, N, K, wrapA || wrapB);
			break;
		default:
			throw std::invalid_argument("NDArray::matmul - unsupported type");
	}
	return c;
}

struct MatmulGeom {
	int batch = 1;
	int M = 0;
	int K = 0;
	int N = 0;
	bool squeezeM = false;
	bool squeezeN = false;
	bool aHasBatch = false;
	bool bHasBatch = false;
};

static MatmulGeom parseGeom(const NDArrayView& a, const NDArrayView& b) {
	int ar = a.getShape().size();
	int br = b.getShape().size();
	MatmulGeom g;

	auto last2 = [](const ArrayList<int>& sh, int& x, int& y) {
		int r = sh.size();
		x = sh.get(r - 2);
		y = sh.get(r - 1);
	};

	if (ar == 1 && br == 1) {
		g.M = 1;
		g.K = a.getShape().get(0);
		g.N = 1;
		g.squeezeM = true;
		g.squeezeN = true;
		if (b.getShape().get(0) != g.K)
			throw std::invalid_argument("NDArray::matmul - inner dimension mismatch");
		return g;
	}
	if (ar == 2 && br == 1) {
		g.M = a.getShape().get(0);
		g.K = a.getShape().get(1);
		g.N = 1;
		g.squeezeN = true;
		if (b.getShape().get(0) != g.K)
			throw std::invalid_argument("NDArray::matmul - inner dimension mismatch");
		return g;
	}
	if (ar == 1 && br == 2) {
		g.M = 1;
		g.K = a.getShape().get(0);
		g.N = b.getShape().get(1);
		g.squeezeM = true;
		if (b.getShape().get(0) != g.K)
			throw std::invalid_argument("NDArray::matmul - inner dimension mismatch");
		return g;
	}
	if (ar == 2 && br == 2) {
		last2(a.getShape(), g.M, g.K);
		int k2;
		last2(b.getShape(), k2, g.N);
		if (k2 != g.K)
			throw std::invalid_argument("NDArray::matmul - inner dimension mismatch");
		return g;
	}
	if (ar == 3 && br == 3) {
		g.batch = a.getShape().get(0);
		if (b.getShape().get(0) != g.batch)
			throw std::invalid_argument("NDArray::matmul - batch mismatch");
		g.M = a.getShape().get(1);
		g.K = a.getShape().get(2);
		if (b.getShape().get(1) != g.K)
			throw std::invalid_argument("NDArray::matmul - inner dimension mismatch");
		g.N = b.getShape().get(2);
		g.aHasBatch = true;
		g.bHasBatch = true;
		return g;
	}
	if (ar == 3 && br == 2) {
		g.batch = a.getShape().get(0);
		g.M = a.getShape().get(1);
		g.K = a.getShape().get(2);
		if (b.getShape().get(0) != g.K)
			throw std::invalid_argument("NDArray::matmul - inner dimension mismatch");
		g.N = b.getShape().get(1);
		g.aHasBatch = true;
		return g;
	}
	if (ar == 2 && br == 3) {
		g.batch = b.getShape().get(0);
		g.M = a.getShape().get(0);
		g.K = a.getShape().get(1);
		if (b.getShape().get(1) != g.K)
			throw std::invalid_argument("NDArray::matmul - inner dimension mismatch");
		g.N = b.getShape().get(2);
		g.bHasBatch = true;
		return g;
	}
	throw std::invalid_argument("NDArray::matmul - rank must be 1, 2, or 3");
}

NDArray ndmatmul(const NDArrayView& a, const NDArrayView& b) {
	if (a.getType() != b.getType())
		throw std::invalid_argument("NDArray::matmul - type mismatch");

	MatmulGeom g = parseGeom(a, b);

	ArrayList<int> outShape;
	if (g.batch > 1)
		outShape.add(g.batch);
	if (!g.squeezeM)
		outShape.add(g.M);
	if (!g.squeezeN)
		outShape.add(g.N);

	if (g.M <= 0 || g.N <= 0 || g.K < 0)
		return NDArray(outShape, matmulResultType(a.getType()));
	if (g.K == 0)
		return NDArray(outShape, matmulResultType(a.getType()));

	bool wrapA = viewIsWrap(a);
	bool wrapB = viewIsWrap(b);

	if (g.batch == 1 && !g.squeezeM && !g.squeezeN) {
		NDArray ad = dense2d(a, g.M, g.K, 1, 0);
		NDArray bd = dense2d(b, g.K, g.N, 1, 0);
		return gemmOwned(ad, bd, wrapA, wrapB);
	}

	NDArray acc(outShape, matmulResultType(a.getType()));
	for (int bi = 0; bi < g.batch; ++bi) {
		int ab = g.aHasBatch ? bi : 0;
		int bb = g.bHasBatch ? bi : 0;
		int aBatchDim = g.aHasBatch ? g.batch : 1;
		int bBatchDim = g.bHasBatch ? g.batch : 1;
		NDArray ad = dense2d(a, g.M, g.K, aBatchDim, ab);
		NDArray bd = dense2d(b, g.K, g.N, bBatchDim, bb);
		NDArray plane = gemmOwned(ad, bd, wrapA, wrapB);

		if (g.batch > 1) {
			storePlane(acc, plane, g.batch, bi);
			continue;
		}
		// squeezed 2D → 1D or scalar
		if (g.squeezeM && g.squeezeN) {
			switch (plane.type) {
				case F32: acc.set({}, plane.get<float>({0, 0})); break;
				case F64: acc.set({}, plane.get<double>({0, 0})); break;
				case UINT256: acc.set({}, plane.get<uint256_t>({0, 0})); break;
				case INT64: acc.set({}, plane.get<int64_t>({0, 0})); break;
				default: acc.set({}, plane.get<int>({0, 0})); break;
			}
			continue;
		}
		int len = g.squeezeM ? g.N : g.M;
		for (int t = 0; t < len; ++t) {
			int i = g.squeezeM ? 0 : t;
			int j = g.squeezeN ? 0 : t;
			switch (plane.type) {
				case F32: acc.set({t}, plane.get<float>({i, j})); break;
				case F64: acc.set({t}, plane.get<double>({i, j})); break;
				case UINT256: acc.set({t}, plane.get<uint256_t>({i, j})); break;
				case INT64: acc.set({t}, plane.get<int64_t>({i, j})); break;
				default: acc.set({t}, plane.get<int>({i, j})); break;
			}
		}
	}
	return acc;
}
