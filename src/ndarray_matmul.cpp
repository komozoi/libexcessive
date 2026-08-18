// Tiled matmul / gemv. Contiguous planes use data() directly (no copy).
// Last two axes are the matrix; leading axis is batch.

#include "ndarray_matmul.h"

#include "NDArray.h"
#include "fs/FdHandle.h"
#include "ndarray_int3.h"
#include "ndarray_half_kernels.h"

#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <thread>

#if defined(_MSC_VER)
#define NDM_RESTRICT __restrict
#else
#define NDM_RESTRICT __restrict__
#endif

#if defined(__AVX512F__)
#define NDM_AVX512 1
#include <immintrin.h>
#else
#define NDM_AVX512 0
#endif

#if defined(__AVX512VNNI__)
#define NDM_VNNI 1
#else
#define NDM_VNNI 0
#endif

#if defined(__AVX512DQ__)
#define NDM_AVX512DQ 1
#else
#define NDM_AVX512DQ 0
#endif

#if defined(__AVX2__)
#define NDM_AVX2 1
#if !NDM_AVX512
#include <immintrin.h>
#endif
#else
#define NDM_AVX2 0
#endif

#if defined(__FMA__) || NDM_AVX512
#define NDM_FMA 1
#else
#define NDM_FMA 0
#endif

#if defined(__AVX512VPOPCNTDQ__)
#define NDM_VPOPCNT 1
#else
#define NDM_VPOPCNT 0
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define NDM_NEON 1
#include <arm_neon.h>
#else
#define NDM_NEON 0
#endif

#if NDM_AVX512
static const int kNR = 16;
static const int kMR = 8;
#else
static const int kNR = 8;
static const int kMR = 8;
#endif
static const int kMC = 72;
static const int kKC = 256;


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
		case F16:
		case BF16:
			return F32;
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
		case F16:
		case BF16:
			return 2;
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

// Spawn only when the GEMM is large enough to hide thread create/join.
static int ndmChooseThreads(size_t work) {
	// std::thread create/join is tens of microseconds; only pay it on huge GEMMs.
	if (work < 80000000ULL)
		return 1;
	unsigned hc = std::thread::hardware_concurrency();
	int n = (int)hc;
	if (n > 8)
		n = 8;
	if (n < 1)
		n = 1;
	return n;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((unused))
#endif
static void ndmRunNStrips(int N, int nrAlign, size_t work,
                          void (*fn)(void*, int, int), void* ctx) {
	int nt = ndmChooseThreads(work);
	int nc = N;
	if (nt > 1) {
		nc = (N + nt - 1) / nt;
		nc = (nc + nrAlign - 1) / nrAlign * nrAlign;
		if (nc < nrAlign)
			nc = nrAlign;
		nt = (N + nc - 1) / nc;
	}
	if (nt <= 1) {
		fn(ctx, 0, N);
		return;
	}
	std::thread* th = new std::thread[nt];
	int t;
	for (t = 0; t < nt; ++t) {
		int ja = t * nc;
		int jb = ja + nc;
		if (ja > N)
			ja = N;
		if (jb > N)
			jb = N;
		th[t] = std::thread(fn, ctx, ja, jb);
	}
	for (t = 0; t < nt; ++t)
		th[t].join();
	delete[] th;
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

static void advisePanel(const void* base, size_t elemBytes, int lda,
                        int row0, int rows, int col0, int cols, bool wrap) {
	if (!wrap || !base || rows <= 0 || cols <= 0 || elemBytes == 0)
		return;
	const char* p = (const char*)base + ((size_t)row0 * (size_t)lda + (size_t)col0) * elemBytes;
	size_t bytes = ((size_t)(rows - 1) * (size_t)lda + (size_t)cols) * elemBytes;
	memoryAdviseWillNeed((void*)p, bytes);
}

// Contiguous packed [rows, cols] (unit last stride; nonzero element offset ok).
static bool isPackedContig2d(const NDArrayView& v, int rows, int cols) {
	if (!v.data() || v.getShape().size() != 2)
		return false;
	if (v.getShape().get(0) != rows || v.getShape().get(1) != cols)
		return false;
	return v.isContiguous();
}

/** First logical element of a contiguous view. Packed types only if offset == 0. */
static const void* contigElemPtr(const NDArrayView& v) {
	const void* base = v.data();
	if (!base)
		return nullptr;
	const size_t es = denseElemSize(v.getType());
	if (es == 0)
		return v.getOffset() == 0 ? base : nullptr;
	return (const char*)base + v.getOffset() * es;
}


#if NDM_AVX2
static inline __m256 ndmFmaddPs(__m256 a, __m256 b, __m256 c) {
#if NDM_FMA
	return _mm256_fmadd_ps(a, b, c);
#else
	return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}
#endif

#if NDM_AVX512
static void kernelF32_8x16(const float* NDM_RESTRICT A, int lda,
                           const float* NDM_RESTRICT B, float* NDM_RESTRICT C, int ldc, int kb) {
	__m512 c0 = _mm512_loadu_ps(C + 0 * ldc);
	__m512 c1 = _mm512_loadu_ps(C + 1 * ldc);
	__m512 c2 = _mm512_loadu_ps(C + 2 * ldc);
	__m512 c3 = _mm512_loadu_ps(C + 3 * ldc);
	__m512 c4 = _mm512_loadu_ps(C + 4 * ldc);
	__m512 c5 = _mm512_loadu_ps(C + 5 * ldc);
	__m512 c6 = _mm512_loadu_ps(C + 6 * ldc);
	__m512 c7 = _mm512_loadu_ps(C + 7 * ldc);
	for (int k = 0; k < kb; ++k) {
		if (k + 8 < kb)
			_mm_prefetch((const char*)(B + (size_t)(k + 8) * (size_t)kNR), _MM_HINT_T0);
		__m512 bv = _mm512_loadu_ps(B + (size_t)k * (size_t)kNR);
		c0 = _mm512_fmadd_ps(_mm512_set1_ps(A[0 * lda + k]), bv, c0);
		c1 = _mm512_fmadd_ps(_mm512_set1_ps(A[1 * lda + k]), bv, c1);
		c2 = _mm512_fmadd_ps(_mm512_set1_ps(A[2 * lda + k]), bv, c2);
		c3 = _mm512_fmadd_ps(_mm512_set1_ps(A[3 * lda + k]), bv, c3);
		c4 = _mm512_fmadd_ps(_mm512_set1_ps(A[4 * lda + k]), bv, c4);
		c5 = _mm512_fmadd_ps(_mm512_set1_ps(A[5 * lda + k]), bv, c5);
		c6 = _mm512_fmadd_ps(_mm512_set1_ps(A[6 * lda + k]), bv, c6);
		c7 = _mm512_fmadd_ps(_mm512_set1_ps(A[7 * lda + k]), bv, c7);
	}
	_mm512_storeu_ps(C + 0 * ldc, c0);
	_mm512_storeu_ps(C + 1 * ldc, c1);
	_mm512_storeu_ps(C + 2 * ldc, c2);
	_mm512_storeu_ps(C + 3 * ldc, c3);
	_mm512_storeu_ps(C + 4 * ldc, c4);
	_mm512_storeu_ps(C + 5 * ldc, c5);
	_mm512_storeu_ps(C + 6 * ldc, c6);
	_mm512_storeu_ps(C + 7 * ldc, c7);
}

static void kernelF32_mx16(const float* NDM_RESTRICT A, int lda,
                           const float* NDM_RESTRICT B, float* NDM_RESTRICT C, int ldc, int kb, int mr) {
	__m512 c[8];
	int r;
	for (r = 0; r < mr; ++r)
		c[r] = _mm512_loadu_ps(C + r * ldc);
	for (int k = 0; k < kb; ++k) {
		__m512 bv = _mm512_loadu_ps(B + (size_t)k * (size_t)kNR);
		for (r = 0; r < mr; ++r)
			c[r] = _mm512_fmadd_ps(_mm512_set1_ps(A[r * lda + k]), bv, c[r]);
	}
	for (r = 0; r < mr; ++r)
		_mm512_storeu_ps(C + r * ldc, c[r]);
}

// Packed B is zero-padded to 16; mask keeps stores inside N.
static void kernelF32_8x16m(const float* NDM_RESTRICT A, int lda,
                            const float* NDM_RESTRICT B, float* NDM_RESTRICT C, int ldc,
                            int kb, int nr) {
	__mmask16 m = (__mmask16)((1u << nr) - 1u);
	__m512 c0 = _mm512_maskz_loadu_ps(m, C + 0 * ldc);
	__m512 c1 = _mm512_maskz_loadu_ps(m, C + 1 * ldc);
	__m512 c2 = _mm512_maskz_loadu_ps(m, C + 2 * ldc);
	__m512 c3 = _mm512_maskz_loadu_ps(m, C + 3 * ldc);
	__m512 c4 = _mm512_maskz_loadu_ps(m, C + 4 * ldc);
	__m512 c5 = _mm512_maskz_loadu_ps(m, C + 5 * ldc);
	__m512 c6 = _mm512_maskz_loadu_ps(m, C + 6 * ldc);
	__m512 c7 = _mm512_maskz_loadu_ps(m, C + 7 * ldc);
	for (int k = 0; k < kb; ++k) {
		__m512 bv = _mm512_loadu_ps(B + (size_t)k * (size_t)kNR);
		c0 = _mm512_fmadd_ps(_mm512_set1_ps(A[0 * lda + k]), bv, c0);
		c1 = _mm512_fmadd_ps(_mm512_set1_ps(A[1 * lda + k]), bv, c1);
		c2 = _mm512_fmadd_ps(_mm512_set1_ps(A[2 * lda + k]), bv, c2);
		c3 = _mm512_fmadd_ps(_mm512_set1_ps(A[3 * lda + k]), bv, c3);
		c4 = _mm512_fmadd_ps(_mm512_set1_ps(A[4 * lda + k]), bv, c4);
		c5 = _mm512_fmadd_ps(_mm512_set1_ps(A[5 * lda + k]), bv, c5);
		c6 = _mm512_fmadd_ps(_mm512_set1_ps(A[6 * lda + k]), bv, c6);
		c7 = _mm512_fmadd_ps(_mm512_set1_ps(A[7 * lda + k]), bv, c7);
	}
	_mm512_mask_storeu_ps(C + 0 * ldc, m, c0);
	_mm512_mask_storeu_ps(C + 1 * ldc, m, c1);
	_mm512_mask_storeu_ps(C + 2 * ldc, m, c2);
	_mm512_mask_storeu_ps(C + 3 * ldc, m, c3);
	_mm512_mask_storeu_ps(C + 4 * ldc, m, c4);
	_mm512_mask_storeu_ps(C + 5 * ldc, m, c5);
	_mm512_mask_storeu_ps(C + 6 * ldc, m, c6);
	_mm512_mask_storeu_ps(C + 7 * ldc, m, c7);
}

static void kernelF32_mx16m(const float* NDM_RESTRICT A, int lda,
                            const float* NDM_RESTRICT B, float* NDM_RESTRICT C, int ldc,
                            int kb, int mr, int nr) {
	__mmask16 m = (__mmask16)((1u << nr) - 1u);
	__m512 c[8];
	int r;
	for (r = 0; r < mr; ++r)
		c[r] = _mm512_maskz_loadu_ps(m, C + r * ldc);
	for (int k = 0; k < kb; ++k) {
		__m512 bv = _mm512_loadu_ps(B + (size_t)k * (size_t)kNR);
		for (r = 0; r < mr; ++r)
			c[r] = _mm512_fmadd_ps(_mm512_set1_ps(A[r * lda + k]), bv, c[r]);
	}
	for (r = 0; r < mr; ++r)
		_mm512_mask_storeu_ps(C + r * ldc, m, c[r]);
}
#endif

#if NDM_AVX2 && !NDM_AVX512
static void kernelF32_8x8(const float* NDM_RESTRICT A, int lda,
                          const float* NDM_RESTRICT B, float* NDM_RESTRICT C, int ldc, int kb) {
	__m256 c0 = _mm256_loadu_ps(C + 0 * ldc);
	__m256 c1 = _mm256_loadu_ps(C + 1 * ldc);
	__m256 c2 = _mm256_loadu_ps(C + 2 * ldc);
	__m256 c3 = _mm256_loadu_ps(C + 3 * ldc);
	__m256 c4 = _mm256_loadu_ps(C + 4 * ldc);
	__m256 c5 = _mm256_loadu_ps(C + 5 * ldc);
	__m256 c6 = _mm256_loadu_ps(C + 6 * ldc);
	__m256 c7 = _mm256_loadu_ps(C + 7 * ldc);
	for (int k = 0; k < kb; ++k) {
		__m256 bv = _mm256_loadu_ps(B + (size_t)k * 8);
		c0 = ndmFmaddPs(_mm256_broadcast_ss(A + 0 * lda + k), bv, c0);
		c1 = ndmFmaddPs(_mm256_broadcast_ss(A + 1 * lda + k), bv, c1);
		c2 = ndmFmaddPs(_mm256_broadcast_ss(A + 2 * lda + k), bv, c2);
		c3 = ndmFmaddPs(_mm256_broadcast_ss(A + 3 * lda + k), bv, c3);
		c4 = ndmFmaddPs(_mm256_broadcast_ss(A + 4 * lda + k), bv, c4);
		c5 = ndmFmaddPs(_mm256_broadcast_ss(A + 5 * lda + k), bv, c5);
		c6 = ndmFmaddPs(_mm256_broadcast_ss(A + 6 * lda + k), bv, c6);
		c7 = ndmFmaddPs(_mm256_broadcast_ss(A + 7 * lda + k), bv, c7);
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
static void kernelF32_8x8_neon(const float* NDM_RESTRICT A, int lda,
                               const float* NDM_RESTRICT B, float* NDM_RESTRICT C, int ldc, int kb) {
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
		const float* bp = B + (size_t)k * 8;
		float32x4_t b0 = vld1q_f32(bp);
		float32x4_t b1 = vld1q_f32(bp + 4);
		float32x4_t a;
		a = vdupq_n_f32(A[0 * lda + k]); c00 = vfmaq_f32(c00, a, b0); c01 = vfmaq_f32(c01, a, b1);
		a = vdupq_n_f32(A[1 * lda + k]); c10 = vfmaq_f32(c10, a, b0); c11 = vfmaq_f32(c11, a, b1);
		a = vdupq_n_f32(A[2 * lda + k]); c20 = vfmaq_f32(c20, a, b0); c21 = vfmaq_f32(c21, a, b1);
		a = vdupq_n_f32(A[3 * lda + k]); c30 = vfmaq_f32(c30, a, b0); c31 = vfmaq_f32(c31, a, b1);
		a = vdupq_n_f32(A[4 * lda + k]); c40 = vfmaq_f32(c40, a, b0); c41 = vfmaq_f32(c41, a, b1);
		a = vdupq_n_f32(A[5 * lda + k]); c50 = vfmaq_f32(c50, a, b0); c51 = vfmaq_f32(c51, a, b1);
		a = vdupq_n_f32(A[6 * lda + k]); c60 = vfmaq_f32(c60, a, b0); c61 = vfmaq_f32(c61, a, b1);
		a = vdupq_n_f32(A[7 * lda + k]); c70 = vfmaq_f32(c70, a, b0); c71 = vfmaq_f32(c71, a, b1);
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

#if !NDM_AVX512
static void kernelF32_scalar(const float* A, int lda, const float* B,
                             float* C, int ldc, int kb, int mr, int nr) {
	for (int i = 0; i < mr; ++i) {
		for (int j = 0; j < nr; ++j) {
			float acc = C[i * ldc + j];
			for (int k = 0; k < kb; ++k)
				acc += A[i * lda + k] * B[(size_t)k * (size_t)kNR + (size_t)j];
			C[i * ldc + j] = acc;
		}
	}
}
#endif

static void packB_f32(const float* B, int ldb, int k0, int kb, int j0, int nb, float* Bp) {
	for (int k = 0; k < kb; ++k) {
		memcpy(Bp + (size_t)k * (size_t)kNR, B + (size_t)(k0 + k) * (size_t)ldb + (size_t)j0,
		       (size_t)nb * sizeof(float));
		if (nb < kNR)
			memset(Bp + (size_t)k * (size_t)kNR + (size_t)nb, 0, (size_t)(kNR - nb) * sizeof(float));
	}
}

static void packA_f32(const float* A, int lda, int i0, int mb, int k0, int kb, float* Ap) {
	for (int i = 0; i < mb; ++i)
		memcpy(Ap + (size_t)i * (size_t)kb, A + (size_t)(i0 + i) * (size_t)lda + (size_t)k0,
		       (size_t)kb * sizeof(float));
}

struct F32Job {
	const float* A;
	const float* B;
	float* C;
	int lda, ldb, ldc, M, N, K;
	bool wrapA, wrapB;
};

static void gemmF32Strip(void* ctx, int j0, int j1) {
	const F32Job* job = (const F32Job*)ctx;
	const float* A = job->A;
	const float* B = job->B;
	float* C = job->C;
	int lda = job->lda;
	int ldb = job->ldb;
	int ldc = job->ldc;
	int M = job->M;
	int K = job->K;
	bool wrapA = job->wrapA;
	bool wrapB = job->wrapB;
	float* Bp = (float*)alloc64((size_t)kKC * (size_t)kNR * sizeof(float));
	float* Ap = (float*)alloc64((size_t)kMC * (size_t)kKC * sizeof(float));

	for (int kc = 0; kc < K; kc += kKC) {
		int kb = K - kc;
		if (kb > kKC)
			kb = kKC;
		int knext = kc + kb;
		if (wrapB && knext < K) {
			int nkb = K - knext;
			if (nkb > kKC)
				nkb = kKC;
			advisePanel(B, 4, ldb, knext, nkb, j0, j1 - j0, true);
		} else
			advisePanel(B, 4, ldb, kc, kb, j0, j1 - j0, wrapB);

		for (int ic = 0; ic < M; ic += kMC) {
			int mb = M - ic;
			if (mb > kMC)
				mb = kMC;
			advisePanel(A, 4, lda, ic, mb, kc, kb, wrapA);
			packA_f32(A, lda, ic, mb, kc, kb, Ap);

			for (int jc = j0; jc < j1; jc += kNR) {
				int nb = j1 - jc;
				if (nb > kNR)
					nb = kNR;
				packB_f32(B, ldb, kc, kb, jc, nb, Bp);

				for (int ir = 0; ir < mb; ) {
					int mr = mb - ir;
					if (mr > kMR)
						mr = kMR;
					float* Cp = C + (size_t)(ic + ir) * (size_t)ldc + (size_t)jc;
					const float* Apanel = Ap + (size_t)ir * (size_t)kb;
#if NDM_AVX512
					if (nb == 16 && mr == 8)
						kernelF32_8x16(Apanel, kb, Bp, Cp, ldc, kb);
					else if (nb == 16)
						kernelF32_mx16(Apanel, kb, Bp, Cp, ldc, kb, mr);
					else if (mr == 8)
						kernelF32_8x16m(Apanel, kb, Bp, Cp, ldc, kb, nb);
					else
						kernelF32_mx16m(Apanel, kb, Bp, Cp, ldc, kb, mr, nb);
#else
#if NDM_AVX2
					if (nb == 8 && mr == 8)
						kernelF32_8x8(Apanel, kb, Bp, Cp, ldc, kb);
					else
#endif
#if NDM_NEON
					if (nb == 8 && mr == 8)
						kernelF32_8x8_neon(Apanel, kb, Bp, Cp, ldc, kb);
					else
#endif
						kernelF32_scalar(Apanel, kb, Bp, Cp, ldc, kb, mr, nb);
#endif
					ir += mr;
				}
			}
		}
	}
	free64(Ap);
	free64(Bp);
}

static void gemmF32(const float* A, int lda, const float* B, int ldb,
                    float* C, int ldc, int M, int N, int K, bool wrapA, bool wrapB) {
	memset(C, 0, (size_t)M * (size_t)ldc * sizeof(float));
	F32Job job;
	job.A = A;
	job.B = B;
	job.C = C;
	job.lda = lda;
	job.ldb = ldb;
	job.ldc = ldc;
	job.M = M;
	job.N = N;
	job.K = K;
	job.wrapA = wrapA;
	job.wrapB = wrapB;
	ndmRunNStrips(N, kNR, (size_t)M * (size_t)N * (size_t)K, gemmF32Strip, &job);
}

enum class NdmHalf { F16, BF16 };

static void packA_half(NdmHalf h, const uint16_t* A, int lda, int i0, int mb, int k0, int kb, float* Ap) {
	for (int i = 0; i < mb; ++i) {
		const uint16_t* row = A + (size_t)(i0 + i) * (size_t)lda + (size_t)k0;
		if (h == NdmHalf::F16)
			ndhalf_f16_to_f32(Ap + (size_t)i * (size_t)kb, row, (size_t)kb);
		else
			ndhalf_bf16_to_f32(Ap + (size_t)i * (size_t)kb, row, (size_t)kb);
	}
}

static void packB_half(NdmHalf h, const uint16_t* B, int ldb, int k0, int kb, int j0, int nb, float* Bp) {
	float tmp[16];
	for (int k = 0; k < kb; ++k) {
		const uint16_t* row = B + (size_t)(k0 + k) * (size_t)ldb + (size_t)j0;
		if (h == NdmHalf::F16)
			ndhalf_f16_to_f32(tmp, row, (size_t)nb);
		else
			ndhalf_bf16_to_f32(tmp, row, (size_t)nb);
		memcpy(Bp + (size_t)k * (size_t)kNR, tmp, (size_t)nb * sizeof(float));
		if (nb < kNR)
			memset(Bp + (size_t)k * (size_t)kNR + (size_t)nb, 0, (size_t)(kNR - nb) * sizeof(float));
	}
}

struct HalfJob {
	NdmHalf half;
	const uint16_t* A;
	const uint16_t* B;
	float* C;
	int lda, ldb, ldc, M, N, K;
	bool wrapA, wrapB;
};

static void gemmHalfStrip(void* ctx, int j0, int j1) {
	const HalfJob* job = (const HalfJob*)ctx;
	const uint16_t* A = job->A;
	const uint16_t* B = job->B;
	float* C = job->C;
	int lda = job->lda;
	int ldb = job->ldb;
	int ldc = job->ldc;
	int M = job->M;
	int K = job->K;
	NdmHalf h = job->half;
	bool wrapA = job->wrapA;
	bool wrapB = job->wrapB;
	float* Bp = (float*)alloc64((size_t)kKC * (size_t)kNR * sizeof(float));
	float* Ap = (float*)alloc64((size_t)kMC * (size_t)kKC * sizeof(float));

	for (int kc = 0; kc < K; kc += kKC) {
		int kb = K - kc;
		if (kb > kKC)
			kb = kKC;
		advisePanel(B, 2, ldb, kc, kb, j0, j1 - j0, wrapB);
		for (int ic = 0; ic < M; ic += kMC) {
			int mb = M - ic;
			if (mb > kMC)
				mb = kMC;
			advisePanel(A, 2, lda, ic, mb, kc, kb, wrapA);
			packA_half(h, A, lda, ic, mb, kc, kb, Ap);
			for (int jc = j0; jc < j1; jc += kNR) {
				int nb = j1 - jc;
				if (nb > kNR)
					nb = kNR;
				packB_half(h, B, ldb, kc, kb, jc, nb, Bp);
				for (int ir = 0; ir < mb; ) {
					int mr = mb - ir;
					if (mr > kMR)
						mr = kMR;
					float* Cp = C + (size_t)(ic + ir) * (size_t)ldc + (size_t)jc;
					const float* Apanel = Ap + (size_t)ir * (size_t)kb;
#if NDM_AVX512
					if (nb == 16 && mr == 8)
						kernelF32_8x16(Apanel, kb, Bp, Cp, ldc, kb);
					else if (nb == 16)
						kernelF32_mx16(Apanel, kb, Bp, Cp, ldc, kb, mr);
					else if (mr == 8)
						kernelF32_8x16m(Apanel, kb, Bp, Cp, ldc, kb, nb);
					else
						kernelF32_mx16m(Apanel, kb, Bp, Cp, ldc, kb, mr, nb);
#else
#if NDM_AVX2
					if (nb == 8 && mr == 8)
						kernelF32_8x8(Apanel, kb, Bp, Cp, ldc, kb);
					else
#endif
#if NDM_NEON
					if (nb == 8 && mr == 8)
						kernelF32_8x8_neon(Apanel, kb, Bp, Cp, ldc, kb);
					else
#endif
						kernelF32_scalar(Apanel, kb, Bp, Cp, ldc, kb, mr, nb);
#endif
					ir += mr;
				}
			}
		}
	}
	free64(Ap);
	free64(Bp);
}

static void gemmHalf(NdmHalf h, const uint16_t* A, int lda, const uint16_t* B, int ldb,
                     float* C, int ldc, int M, int N, int K, bool wrapA, bool wrapB) {
	memset(C, 0, (size_t)M * (size_t)ldc * sizeof(float));
	HalfJob job;
	job.half = h;
	job.A = A;
	job.B = B;
	job.C = C;
	job.lda = lda;
	job.ldb = ldb;
	job.ldc = ldc;
	job.M = M;
	job.N = N;
	job.K = K;
	job.wrapA = wrapA;
	job.wrapB = wrapB;
	ndmRunNStrips(N, kNR, (size_t)M * (size_t)N * (size_t)K, gemmHalfStrip, &job);
}

static void gemvHalf(NdmHalf h, const uint16_t* A, int lda, const uint16_t* x, float* y, int M, int K) {
	for (int i = 0; i < M; ++i) {
		const uint16_t* row = A + (size_t)i * (size_t)lda;
		y[i] = (h == NdmHalf::F16) ? ndhalf_dot_f16(row, x, (size_t)K)
		                           : ndhalf_dot_bf16(row, x, (size_t)K);
	}
}


#if NDM_AVX512
static void kernelF64_8x8(const double* A, int lda, const double* B, double* C, int ldc, int kb) {
	__m512d c0 = _mm512_loadu_pd(C + 0 * ldc);
	__m512d c1 = _mm512_loadu_pd(C + 1 * ldc);
	__m512d c2 = _mm512_loadu_pd(C + 2 * ldc);
	__m512d c3 = _mm512_loadu_pd(C + 3 * ldc);
	__m512d c4 = _mm512_loadu_pd(C + 4 * ldc);
	__m512d c5 = _mm512_loadu_pd(C + 5 * ldc);
	__m512d c6 = _mm512_loadu_pd(C + 6 * ldc);
	__m512d c7 = _mm512_loadu_pd(C + 7 * ldc);
	for (int k = 0; k < kb; ++k) {
		__m512d bv = _mm512_loadu_pd(B + (size_t)k * 8);
		c0 = _mm512_fmadd_pd(_mm512_set1_pd(A[0 * lda + k]), bv, c0);
		c1 = _mm512_fmadd_pd(_mm512_set1_pd(A[1 * lda + k]), bv, c1);
		c2 = _mm512_fmadd_pd(_mm512_set1_pd(A[2 * lda + k]), bv, c2);
		c3 = _mm512_fmadd_pd(_mm512_set1_pd(A[3 * lda + k]), bv, c3);
		c4 = _mm512_fmadd_pd(_mm512_set1_pd(A[4 * lda + k]), bv, c4);
		c5 = _mm512_fmadd_pd(_mm512_set1_pd(A[5 * lda + k]), bv, c5);
		c6 = _mm512_fmadd_pd(_mm512_set1_pd(A[6 * lda + k]), bv, c6);
		c7 = _mm512_fmadd_pd(_mm512_set1_pd(A[7 * lda + k]), bv, c7);
	}
	_mm512_storeu_pd(C + 0 * ldc, c0);
	_mm512_storeu_pd(C + 1 * ldc, c1);
	_mm512_storeu_pd(C + 2 * ldc, c2);
	_mm512_storeu_pd(C + 3 * ldc, c3);
	_mm512_storeu_pd(C + 4 * ldc, c4);
	_mm512_storeu_pd(C + 5 * ldc, c5);
	_mm512_storeu_pd(C + 6 * ldc, c6);
	_mm512_storeu_pd(C + 7 * ldc, c7);
}
#endif

static void gemmF64(const double* A, int lda, const double* B, int ldb,
                    double* C, int ldc, int M, int N, int K, bool wrapA, bool wrapB) {
	memset(C, 0, (size_t)M * (size_t)ldc * sizeof(double));
	const int nr = 8;
	double* Bp = (double*)alloc64((size_t)kKC * (size_t)nr * sizeof(double));
	double* Ap = (double*)alloc64((size_t)kMC * (size_t)kKC * sizeof(double));
	for (int kc = 0; kc < K; kc += kKC) {
		int kb = K - kc;
		if (kb > kKC)
			kb = kKC;
		advisePanel(B, 8, ldb, kc, kb, 0, N, wrapB);
		for (int jc = 0; jc < N; jc += nr) {
			int nb = N - jc;
			if (nb > nr)
				nb = nr;
			for (int k = 0; k < kb; ++k) {
				memcpy(Bp + (size_t)k * (size_t)nr, B + (size_t)(kc + k) * (size_t)ldb + (size_t)jc,
				       (size_t)nb * sizeof(double));
				if (nb < nr)
					memset(Bp + (size_t)k * (size_t)nr + (size_t)nb, 0, (size_t)(nr - nb) * sizeof(double));
			}
			for (int ic = 0; ic < M; ic += kMC) {
				int mb = M - ic;
				if (mb > kMC)
					mb = kMC;
				advisePanel(A, 8, lda, ic, mb, kc, kb, wrapA);
				for (int i = 0; i < mb; ++i)
					memcpy(Ap + (size_t)i * (size_t)kb,
					       A + (size_t)(ic + i) * (size_t)lda + (size_t)kc, (size_t)kb * sizeof(double));
				for (int ir = 0; ir < mb; ) {
					int mr = mb - ir;
					if (mr > 8)
						mr = 8;
					double* Cp = C + (size_t)(ic + ir) * (size_t)ldc + (size_t)jc;
#if NDM_AVX512
					if (nb == 8 && mr == 8) {
						kernelF64_8x8(Ap + (size_t)ir * (size_t)kb, kb, Bp, Cp, ldc, kb);
						ir += mr;
						continue;
					}
#endif
					for (int i = 0; i < mr; ++i) {
						for (int j = 0; j < nb; ++j) {
							double acc = Cp[i * ldc + j];
							const double* ar = Ap + (size_t)(ir + i) * (size_t)kb;
							for (int k = 0; k < kb; ++k)
								acc += ar[k] * Bp[(size_t)k * (size_t)nr + (size_t)j];
							Cp[i * ldc + j] = acc;
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


#if NDM_VNNI
static const int kI8NR = 16;
static const int kI8MR = 16;
static const int kI8MC = 96;
static const int kI8KC_LO = 256;
static const int kI8KC_HI = 512;

// xorB 0x80 is B-128 as int8 (UINT8 path). xorB 0 leaves bits unchanged.
static void packB_vnni(const uint8_t* B, int ldb, int k0, int kb, int j0, int nb, int8_t* Bp,
                       uint8_t xorB) {
	int kb4 = (kb + 3) & ~3;
	memset(Bp, 0, (size_t)((kb4 / 4) * 64));
	__m128i x = _mm_set1_epi8((char)xorB);
	int k;
	for (k = 0; k + 4 <= kb; k += 4) {
		int8_t* out = Bp + ((size_t)k / 4) * 64;
		if (nb == 16) {
			const uint8_t* r0 = B + (size_t)(k0 + k + 0) * (size_t)ldb + (size_t)j0;
			const uint8_t* r1 = B + (size_t)(k0 + k + 1) * (size_t)ldb + (size_t)j0;
			const uint8_t* r2 = B + (size_t)(k0 + k + 2) * (size_t)ldb + (size_t)j0;
			const uint8_t* r3 = B + (size_t)(k0 + k + 3) * (size_t)ldb + (size_t)j0;
			__m128i v0 = _mm_xor_si128(_mm_loadu_si128((const __m128i*)r0), x);
			__m128i v1 = _mm_xor_si128(_mm_loadu_si128((const __m128i*)r1), x);
			__m128i v2 = _mm_xor_si128(_mm_loadu_si128((const __m128i*)r2), x);
			__m128i v3 = _mm_xor_si128(_mm_loadu_si128((const __m128i*)r3), x);
			__m128i a = _mm_unpacklo_epi8(v0, v1);
			__m128i b = _mm_unpackhi_epi8(v0, v1);
			__m128i c = _mm_unpacklo_epi8(v2, v3);
			__m128i d = _mm_unpackhi_epi8(v2, v3);
			_mm_storeu_si128((__m128i*)(out + 0), _mm_unpacklo_epi16(a, c));
			_mm_storeu_si128((__m128i*)(out + 16), _mm_unpackhi_epi16(a, c));
			_mm_storeu_si128((__m128i*)(out + 32), _mm_unpacklo_epi16(b, d));
			_mm_storeu_si128((__m128i*)(out + 48), _mm_unpackhi_epi16(b, d));
		} else {
			for (int j = 0; j < nb; ++j) {
				out[j * 4 + 0] = (int8_t)(B[(size_t)(k0 + k + 0) * (size_t)ldb + (size_t)(j0 + j)] ^ xorB);
				out[j * 4 + 1] = (int8_t)(B[(size_t)(k0 + k + 1) * (size_t)ldb + (size_t)(j0 + j)] ^ xorB);
				out[j * 4 + 2] = (int8_t)(B[(size_t)(k0 + k + 2) * (size_t)ldb + (size_t)(j0 + j)] ^ xorB);
				out[j * 4 + 3] = (int8_t)(B[(size_t)(k0 + k + 3) * (size_t)ldb + (size_t)(j0 + j)] ^ xorB);
			}
		}
	}
	if (k < kb) {
		int8_t* out = Bp + ((size_t)k / 4) * 64;
		for (int j = 0; j < nb; ++j) {
			for (int t = 0; k + t < kb; ++t)
				out[j * 4 + t] = (int8_t)(B[(size_t)(k0 + k + t) * (size_t)ldb + (size_t)(j0 + j)] ^ xorB);
		}
	}
}

static void kernelI8_16x16(const uint8_t* A, int lda, const int8_t* Bp, int32_t* C, int ldc, int kb, int mr, int nr) {
	__m512i c0 = _mm512_setzero_si512();
	__m512i c1 = _mm512_setzero_si512();
	__m512i c2 = _mm512_setzero_si512();
	__m512i c3 = _mm512_setzero_si512();
	__m512i c4 = _mm512_setzero_si512();
	__m512i c5 = _mm512_setzero_si512();
	__m512i c6 = _mm512_setzero_si512();
	__m512i c7 = _mm512_setzero_si512();
	__m512i c8 = _mm512_setzero_si512();
	__m512i c9 = _mm512_setzero_si512();
	__m512i c10 = _mm512_setzero_si512();
	__m512i c11 = _mm512_setzero_si512();
	__m512i c12 = _mm512_setzero_si512();
	__m512i c13 = _mm512_setzero_si512();
	__m512i c14 = _mm512_setzero_si512();
	__m512i c15 = _mm512_setzero_si512();
	int k = 0;
	if (mr == 16) {
		const uint8_t* a0 = A + 0 * (size_t)lda;
		const uint8_t* a1 = A + 1 * (size_t)lda;
		const uint8_t* a2 = A + 2 * (size_t)lda;
		const uint8_t* a3 = A + 3 * (size_t)lda;
		const uint8_t* a4 = A + 4 * (size_t)lda;
		const uint8_t* a5 = A + 5 * (size_t)lda;
		const uint8_t* a6 = A + 6 * (size_t)lda;
		const uint8_t* a7 = A + 7 * (size_t)lda;
		const uint8_t* a8 = A + 8 * (size_t)lda;
		const uint8_t* a9 = A + 9 * (size_t)lda;
		const uint8_t* a10 = A + 10 * (size_t)lda;
		const uint8_t* a11 = A + 11 * (size_t)lda;
		const uint8_t* a12 = A + 12 * (size_t)lda;
		const uint8_t* a13 = A + 13 * (size_t)lda;
		const uint8_t* a14 = A + 14 * (size_t)lda;
		const uint8_t* a15 = A + 15 * (size_t)lda;
		const int8_t* bp = Bp;
		for (; k + 4 <= kb; k += 4) {
			__m512i bv = _mm512_loadu_si512((const void*)bp);
			bp += 64;
			c0 = _mm512_dpbusd_epi32(c0, _mm512_set1_epi32(*(const int*)(a0 + k)), bv);
			c1 = _mm512_dpbusd_epi32(c1, _mm512_set1_epi32(*(const int*)(a1 + k)), bv);
			c2 = _mm512_dpbusd_epi32(c2, _mm512_set1_epi32(*(const int*)(a2 + k)), bv);
			c3 = _mm512_dpbusd_epi32(c3, _mm512_set1_epi32(*(const int*)(a3 + k)), bv);
			c4 = _mm512_dpbusd_epi32(c4, _mm512_set1_epi32(*(const int*)(a4 + k)), bv);
			c5 = _mm512_dpbusd_epi32(c5, _mm512_set1_epi32(*(const int*)(a5 + k)), bv);
			c6 = _mm512_dpbusd_epi32(c6, _mm512_set1_epi32(*(const int*)(a6 + k)), bv);
			c7 = _mm512_dpbusd_epi32(c7, _mm512_set1_epi32(*(const int*)(a7 + k)), bv);
			c8 = _mm512_dpbusd_epi32(c8, _mm512_set1_epi32(*(const int*)(a8 + k)), bv);
			c9 = _mm512_dpbusd_epi32(c9, _mm512_set1_epi32(*(const int*)(a9 + k)), bv);
			c10 = _mm512_dpbusd_epi32(c10, _mm512_set1_epi32(*(const int*)(a10 + k)), bv);
			c11 = _mm512_dpbusd_epi32(c11, _mm512_set1_epi32(*(const int*)(a11 + k)), bv);
			c12 = _mm512_dpbusd_epi32(c12, _mm512_set1_epi32(*(const int*)(a12 + k)), bv);
			c13 = _mm512_dpbusd_epi32(c13, _mm512_set1_epi32(*(const int*)(a13 + k)), bv);
			c14 = _mm512_dpbusd_epi32(c14, _mm512_set1_epi32(*(const int*)(a14 + k)), bv);
			c15 = _mm512_dpbusd_epi32(c15, _mm512_set1_epi32(*(const int*)(a15 + k)), bv);
		}
	} else {
		const int8_t* bp = Bp;
		__m512i acc[16];
		int r;
		for (r = 0; r < 16; ++r)
			acc[r] = _mm512_setzero_si512();
		for (; k + 4 <= kb; k += 4) {
			__m512i bv = _mm512_loadu_si512((const void*)bp);
			bp += 64;
			for (r = 0; r < mr; ++r) {
				int av = 0;
				memcpy(&av, A + (size_t)r * (size_t)lda + (size_t)k, 4);
				acc[r] = _mm512_dpbusd_epi32(acc[r], _mm512_set1_epi32(av), bv);
			}
		}
		c0 = acc[0]; c1 = acc[1]; c2 = acc[2]; c3 = acc[3];
		c4 = acc[4]; c5 = acc[5]; c6 = acc[6]; c7 = acc[7];
		c8 = acc[8]; c9 = acc[9]; c10 = acc[10]; c11 = acc[11];
		c12 = acc[12]; c13 = acc[13]; c14 = acc[14]; c15 = acc[15];
	}

	alignas(64) int32_t tmp[16][16];
	_mm512_store_si512(tmp[0], c0);
	_mm512_store_si512(tmp[1], c1);
	_mm512_store_si512(tmp[2], c2);
	_mm512_store_si512(tmp[3], c3);
	_mm512_store_si512(tmp[4], c4);
	_mm512_store_si512(tmp[5], c5);
	_mm512_store_si512(tmp[6], c6);
	_mm512_store_si512(tmp[7], c7);
	_mm512_store_si512(tmp[8], c8);
	_mm512_store_si512(tmp[9], c9);
	_mm512_store_si512(tmp[10], c10);
	_mm512_store_si512(tmp[11], c11);
	_mm512_store_si512(tmp[12], c12);
	_mm512_store_si512(tmp[13], c13);
	_mm512_store_si512(tmp[14], c14);
	_mm512_store_si512(tmp[15], c15);
	for (int r = 0; r < mr; ++r) {
		int32_t* crow = C + r * ldc;
		for (int j = 0; j < nr; ++j)
			crow[j] += tmp[r][j];
		for (int kk = k; kk < kb; ++kk) {
			int a = (int)A[(size_t)r * (size_t)lda + (size_t)kk];
			for (int j = 0; j < nr; ++j)
				crow[j] += a * (int)Bp[((size_t)(kk / 4) * 64) + (size_t)j * 4 + (size_t)(kk % 4)];
		}
	}
}

// A,B uint8. VPDPBUSD is u8×i8, so B is stored as int8 (same bits 0..127;
// values 128..255 are negative — we only use this for INT3 0..7 and UINT8 via
// B-128 + correction, or INT8 via A+128 + correction).
struct VNNIJob {
	const uint8_t* A;
	const int8_t* B;
	int32_t* C;
	int lda, ldb, ldc, M, N, K;
	uint8_t xorA, xorB;
};

static void gemmVNNIStrip(void* ctx, int j0, int j1) {
	const VNNIJob* job = (const VNNIJob*)ctx;
	const uint8_t* A = job->A;
	const int8_t* B = job->B;
	int32_t* C = job->C;
	int lda = job->lda;
	int ldb = job->ldb;
	int ldc = job->ldc;
	int M = job->M;
	int K = job->K;
	uint8_t xorA = job->xorA;
	uint8_t xorB = job->xorB;
	int nb = j1 - j0;
	if (nb <= 0)
		return;
	int kcStep = (K > 1024) ? kI8KC_HI : kI8KC_LO;
	if (kcStep > K)
		kcStep = K;
	int nPanels = (nb + kI8NR - 1) / kI8NR;
	size_t panelBytes = ((size_t)((kcStep + 3) / 4)) * 64;
	int8_t* Bp = (int8_t*)alloc64((size_t)nPanels * panelBytes);
	uint8_t* Ap = (uint8_t*)alloc64((size_t)kI8MC * (size_t)kcStep);
	for (int kc = 0; kc < K; kc += kcStep) {
		int kb = K - kc;
		if (kb > kcStep)
			kb = kcStep;
		for (int p = 0; p < nPanels; ++p) {
			int jr = p * kI8NR;
			int nr = nb - jr;
			if (nr > kI8NR)
				nr = kI8NR;
			packB_vnni((const uint8_t*)B, ldb, kc, kb, j0 + jr, nr,
			           Bp + (size_t)p * panelBytes, xorB);
		}
		for (int ic = 0; ic < M; ic += kI8MC) {
			int mb = M - ic;
			if (mb > kI8MC)
				mb = kI8MC;
			for (int i = 0; i < mb; ++i) {
				const uint8_t* src = A + (size_t)(ic + i) * (size_t)lda + (size_t)kc;
				uint8_t* dst = Ap + (size_t)i * (size_t)kb;
				if (xorA == 0)
					memcpy(dst, src, (size_t)kb);
				else {
					int k = 0;
					for (; k + 16 <= kb; k += 16) {
						__m128i v = _mm_loadu_si128((const __m128i*)(src + k));
						_mm_storeu_si128((__m128i*)(dst + k),
						                 _mm_xor_si128(v, _mm_set1_epi8((char)xorA)));
					}
					for (; k < kb; ++k)
						dst[k] = (uint8_t)(src[k] ^ xorA);
				}
			}
			for (int ir = 0; ir < mb; ir += kI8MR) {
				int mr = mb - ir;
				if (mr > kI8MR)
					mr = kI8MR;
				for (int p = 0; p < nPanels; ++p) {
					int jr = p * kI8NR;
					int nr = nb - jr;
					if (nr > kI8NR)
						nr = kI8NR;
					kernelI8_16x16(Ap + (size_t)ir * (size_t)kb, kb,
					               Bp + (size_t)p * panelBytes,
					               C + (size_t)(ic + ir) * (size_t)ldc + (size_t)(j0 + jr),
					               ldc, kb, mr, nr);
				}
			}
		}
	}
	free64(Ap);
	free64(Bp);
}

static void gemmVNNI_u8i8(const uint8_t* A, int lda, const int8_t* B, int ldb,
                          int32_t* C, int ldc, int M, int N, int K,
                          uint8_t xorA, uint8_t xorB) {
	VNNIJob job;
	job.A = A;
	job.B = B;
	job.C = C;
	job.lda = lda;
	job.ldb = ldb;
	job.ldc = ldc;
	job.M = M;
	job.N = N;
	job.K = K;
	job.xorA = xorA;
	job.xorB = xorB;
	ndmRunNStrips(N, kI8NR, (size_t)M * (size_t)N * (size_t)K, gemmVNNIStrip, &job);
}
#endif

#if !NDM_VNNI
static void gemmI8_scalar(const int8_t* A, int lda, const int8_t* B, int ldb,
                          int32_t* C, int ldc, int M, int N, int K) {
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			int32_t acc = C[i * ldc + j];
			for (int k = 0; k < K; ++k)
				acc += (int32_t)A[i * lda + k] * (int32_t)B[k * ldb + j];
			C[i * ldc + j] = acc;
		}
	}
}

static void gemmU8_scalar(const uint8_t* A, int lda, const uint8_t* B, int ldb,
                          int32_t* C, int ldc, int M, int N, int K) {
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			int32_t acc = C[i * ldc + j];
			for (int k = 0; k < K; ++k)
				acc += (int32_t)A[i * lda + k] * (int32_t)B[k * ldb + j];
			C[i * ldc + j] = acc;
		}
	}
}
#endif

static void gemmINT8(const int8_t* A, int lda, const int8_t* B, int ldb,
                     int32_t* C, int ldc, int M, int N, int K) {
	memset(C, 0, (size_t)M * (size_t)ldc * sizeof(int32_t));
#if NDM_VNNI
	int32_t* col = (int32_t*)calloc((size_t)N, sizeof(int32_t));
	if (!col)
		throw std::bad_alloc();
	for (int k = 0; k < K; ++k)
		for (int j = 0; j < N; ++j)
			col[j] += (int32_t)B[(size_t)k * (size_t)ldb + (size_t)j];
	gemmVNNI_u8i8((const uint8_t*)A, lda, B, ldb, C, ldc, M, N, K, 0x80, 0);
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
			C[i * ldc + j] -= 128 * col[j];
	free(col);
#else
	gemmI8_scalar(A, lda, B, ldb, C, ldc, M, N, K);
#endif
}

static void gemmUINT8(const uint8_t* A, int lda, const uint8_t* B, int ldb,
                      int32_t* C, int ldc, int M, int N, int K) {
	memset(C, 0, (size_t)M * (size_t)ldc * sizeof(int32_t));
#if NDM_VNNI
	int32_t* row = (int32_t*)calloc((size_t)M, sizeof(int32_t));
	if (!row)
		throw std::bad_alloc();
	for (int i = 0; i < M; ++i) {
		int32_t s = 0;
		for (int k = 0; k < K; ++k)
			s += (int32_t)A[(size_t)i * (size_t)lda + (size_t)k];
		row[i] = s;
	}
	gemmVNNI_u8i8(A, lda, (const int8_t*)B, ldb, C, ldc, M, N, K, 0, 0x80);
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
			C[i * ldc + j] += 128 * row[i];
	free(row);
#else
	gemmU8_scalar(A, lda, B, ldb, C, ldc, M, N, K);
#endif
}

static void unpackInt3Xor4(uint8_t* dst, const uint64_t* src, size_t n) {
	size_t words = n / 16;
	size_t w = 0;
#if NDM_AVX512
	for (; w + 4 <= words; w += 4) {
		__m256i p = _mm256_loadu_si256((const __m256i*)(src + w));
		__m256i lo = _mm256_and_si256(p, _mm256_set1_epi8(0x07));
		__m256i hi = _mm256_and_si256(_mm256_srli_epi16(p, 4), _mm256_set1_epi8(0x07));
		lo = _mm256_xor_si256(lo, _mm256_set1_epi8(0x04));
		hi = _mm256_xor_si256(hi, _mm256_set1_epi8(0x04));
		__m128i lo0 = _mm256_castsi256_si128(lo);
		__m128i lo1 = _mm256_extracti128_si256(lo, 1);
		__m128i hi0 = _mm256_castsi256_si128(hi);
		__m128i hi1 = _mm256_extracti128_si256(hi, 1);
		__m256i r0 = _mm256_set_m128i(_mm_unpackhi_epi8(lo0, hi0), _mm_unpacklo_epi8(lo0, hi0));
		__m256i r1 = _mm256_set_m128i(_mm_unpackhi_epi8(lo1, hi1), _mm_unpacklo_epi8(lo1, hi1));
		_mm256_storeu_si256((__m256i*)(dst + w * 16), r0);
		_mm256_storeu_si256((__m256i*)(dst + w * 16 + 32), r1);
	}
#endif
	for (; w < words; ++w) {
		uint64_t x = (src[w] & int3_kValueMask) ^ 0x4444444444444444ULL;
		for (int lane = 0; lane < 16; ++lane)
			dst[w * 16 + (size_t)lane] = (uint8_t)((x >> (unsigned)(lane * 4)) & 7u);
	}
	for (size_t i = words * 16; i < n; ++i)
		dst[i] = (uint8_t)(int3_getSigned(src, i) + 4);
}

#if NDM_AVX512
static inline __m512i ndmNibbleWordToEpi32(uint64_t packed) {
	__m128i b = _mm_cvtsi64_si128((long long)(packed & 0x7777777777777777ULL));
	__m128i lo = _mm_and_si128(b, _mm_set1_epi8(0x0f));
	__m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(0x0f));
	__m128i nib = _mm_unpacklo_epi8(lo, hi);
	__m128i ge4 = _mm_cmpgt_epi8(nib, _mm_set1_epi8(3));
	nib = _mm_add_epi8(nib, _mm_and_si128(ge4, _mm_set1_epi8((char)-8)));
	return _mm512_cvtepi8_epi32(nib);
}
#endif

static void unpackInt3SignedI32(int32_t* dst, const uint64_t* src, size_t n) {
	size_t words = n / 16;
	size_t w = 0;
#if NDM_AVX512
	for (; w < words; ++w)
		_mm512_storeu_si512(dst + w * 16, ndmNibbleWordToEpi32(src[w]));
#else
	for (; w < words; ++w) {
		uint64_t p = src[w];
		for (int lane = 0; lane < 16; ++lane) {
			int v = (int)((p >> (unsigned)(lane * 4)) & 7u);
			dst[w * 16 + (size_t)lane] = (v >= 4) ? v - 8 : v;
		}
	}
#endif
	for (size_t i = words * 16; i < n; ++i)
		dst[i] = int3_getSigned(src, i);
}

struct I3StreamJob {
	int32_t* C;
	const int32_t* X;
	const uint64_t* Bw;
	int ldc;
	int M;
	int N;
	int K;
};

static void gemmINT3_streamB_strip(void* ctx, int j0, int j1) {
	const I3StreamJob* job = (const I3StreamJob*)ctx;
	int32_t* C = job->C;
	const int32_t* X = job->X;
	const uint64_t* Bw = job->Bw;
	int ldc = job->ldc;
	int M = job->M;
	int N = job->N;
	int K = job->K;
	int wordsN = N / 16;
	int w0 = j0 / 16;
	int w1 = j1 / 16;
	if (w1 <= w0)
		return;

#if NDM_AVX512
	if (M == 1) {
		int width = (w1 - w0) * 16;
		memset(C + j0, 0, (size_t)width * sizeof(int32_t));
		for (int k = 0; k < K; ++k) {
			const uint64_t* row = Bw + (size_t)k * (size_t)wordsN + (size_t)w0;
			__m512i vx = _mm512_set1_epi32(X[k]);
			int32_t* yy = C + j0;
			for (int ww = 0; ww < w1 - w0; ++ww) {
				__m512i acc = _mm512_loadu_si512(yy + ww * 16);
				acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(vx, ndmNibbleWordToEpi32(row[ww])));
				_mm512_storeu_si512(yy + ww * 16, acc);
			}
		}
	} else {
		int m0 = 0;
		for (; m0 + 8 <= M; m0 += 8) {
			for (int w = w0; w < w1; ++w) {
				__m512i a0 = _mm512_setzero_si512();
				__m512i a1 = _mm512_setzero_si512();
				__m512i a2 = _mm512_setzero_si512();
				__m512i a3 = _mm512_setzero_si512();
				__m512i a4 = _mm512_setzero_si512();
				__m512i a5 = _mm512_setzero_si512();
				__m512i a6 = _mm512_setzero_si512();
				__m512i a7 = _mm512_setzero_si512();
				const int32_t* x0 = X + (size_t)(m0 + 0) * (size_t)K;
				const int32_t* x1 = X + (size_t)(m0 + 1) * (size_t)K;
				const int32_t* x2 = X + (size_t)(m0 + 2) * (size_t)K;
				const int32_t* x3 = X + (size_t)(m0 + 3) * (size_t)K;
				const int32_t* x4 = X + (size_t)(m0 + 4) * (size_t)K;
				const int32_t* x5 = X + (size_t)(m0 + 5) * (size_t)K;
				const int32_t* x6 = X + (size_t)(m0 + 6) * (size_t)K;
				const int32_t* x7 = X + (size_t)(m0 + 7) * (size_t)K;
				for (int k = 0; k < K; ++k) {
					__m512i vw = ndmNibbleWordToEpi32(Bw[(size_t)k * (size_t)wordsN + (size_t)w]);
					a0 = _mm512_add_epi32(a0, _mm512_mullo_epi32(_mm512_set1_epi32(x0[k]), vw));
					a1 = _mm512_add_epi32(a1, _mm512_mullo_epi32(_mm512_set1_epi32(x1[k]), vw));
					a2 = _mm512_add_epi32(a2, _mm512_mullo_epi32(_mm512_set1_epi32(x2[k]), vw));
					a3 = _mm512_add_epi32(a3, _mm512_mullo_epi32(_mm512_set1_epi32(x3[k]), vw));
					a4 = _mm512_add_epi32(a4, _mm512_mullo_epi32(_mm512_set1_epi32(x4[k]), vw));
					a5 = _mm512_add_epi32(a5, _mm512_mullo_epi32(_mm512_set1_epi32(x5[k]), vw));
					a6 = _mm512_add_epi32(a6, _mm512_mullo_epi32(_mm512_set1_epi32(x6[k]), vw));
					a7 = _mm512_add_epi32(a7, _mm512_mullo_epi32(_mm512_set1_epi32(x7[k]), vw));
				}
				int j = w * 16;
				_mm512_storeu_si512(C + (size_t)(m0 + 0) * (size_t)ldc + (size_t)j, a0);
				_mm512_storeu_si512(C + (size_t)(m0 + 1) * (size_t)ldc + (size_t)j, a1);
				_mm512_storeu_si512(C + (size_t)(m0 + 2) * (size_t)ldc + (size_t)j, a2);
				_mm512_storeu_si512(C + (size_t)(m0 + 3) * (size_t)ldc + (size_t)j, a3);
				_mm512_storeu_si512(C + (size_t)(m0 + 4) * (size_t)ldc + (size_t)j, a4);
				_mm512_storeu_si512(C + (size_t)(m0 + 5) * (size_t)ldc + (size_t)j, a5);
				_mm512_storeu_si512(C + (size_t)(m0 + 6) * (size_t)ldc + (size_t)j, a6);
				_mm512_storeu_si512(C + (size_t)(m0 + 7) * (size_t)ldc + (size_t)j, a7);
			}
		}
		int width = (w1 - w0) * 16;
		for (int m = m0; m < M; ++m)
			memset(C + (size_t)m * (size_t)ldc + (size_t)j0, 0, (size_t)width * sizeof(int32_t));
		for (int k = 0; k < K; ++k) {
			const uint64_t* row = Bw + (size_t)k * (size_t)wordsN + (size_t)w0;
			for (int m = m0; m < M; ++m) {
				__m512i vx = _mm512_set1_epi32(X[(size_t)m * (size_t)K + (size_t)k]);
				int32_t* yy = C + (size_t)m * (size_t)ldc + (size_t)j0;
				for (int ww = 0; ww < w1 - w0; ++ww) {
					__m512i acc = _mm512_loadu_si512(yy + ww * 16);
					acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(vx, ndmNibbleWordToEpi32(row[ww])));
					_mm512_storeu_si512(yy + ww * 16, acc);
				}
			}
		}
	}
#else
	int width = (w1 - w0) * 16;
	for (int m = 0; m < M; ++m)
		memset(C + (size_t)m * (size_t)ldc + (size_t)j0, 0, (size_t)width * sizeof(int32_t));
	for (int k = 0; k < K; ++k) {
		const uint64_t* row = Bw + (size_t)k * (size_t)wordsN;
		for (int w = w0; w < w1; ++w) {
			uint64_t packed = row[w];
			int jbase = w * 16;
			for (int m = 0; m < M; ++m) {
				int32_t xk = X[(size_t)m * (size_t)K + (size_t)k];
				int32_t* yy = C + (size_t)m * (size_t)ldc + (size_t)jbase;
				for (int t = 0; t < 16; ++t) {
					int v = (int)((packed >> (unsigned)(t * 4)) & 7u);
					yy[t] += xk * ((v >= 4) ? v - 8 : v);
				}
			}
		}
	}
#endif
}

static void gemmINT3_streamB(const int32_t* X, const uint64_t* Bw, int32_t* C, int ldc,
                             int M, int N, int K) {
	I3StreamJob job;
	job.C = C;
	job.X = X;
	job.Bw = Bw;
	job.ldc = ldc;
	job.M = M;
	job.N = N;
	job.K = K;
	size_t work = 0;
	if (M >= 16 && (size_t)K * (size_t)N >= 8000000ull)
		work = (size_t)M * (size_t)N * (size_t)K;
	ndmRunNStrips(N, 16, work, gemmINT3_streamB_strip, &job);
}

static void gemvINT3_streamA(const uint64_t* Aw, const int32_t* x, int32_t* y, int M, int K) {
	int wordsK = K / 16;
	for (int i = 0; i < M; ++i) {
		const uint64_t* row = Aw + ((size_t)i * (size_t)K) / 16;
#if NDM_AVX512
		__m512i accv = _mm512_setzero_si512();
		int w = 0;
		for (; w < wordsK; ++w) {
			__m512i av = ndmNibbleWordToEpi32(row[w]);
			__m512i xv = _mm512_loadu_si512(x + w * 16);
			accv = _mm512_add_epi32(accv, _mm512_mullo_epi32(av, xv));
		}
		int32_t acc = (int32_t)_mm512_reduce_add_epi32(accv);
#else
		int32_t acc = 0;
		int w = 0;
		for (; w < wordsK; ++w) {
			uint64_t packed = row[w];
			for (int t = 0; t < 16; ++t) {
				int v = (int)((packed >> (unsigned)(t * 4)) & 7u);
				acc += ((v >= 4) ? v - 8 : v) * x[w * 16 + t];
			}
		}
#endif
		for (int k = wordsK * 16; k < K; ++k)
			acc += int3_getSigned(Aw, (size_t)i * (size_t)K + (size_t)k) * x[k];
		y[i] = acc;
	}
}

#if NDM_VNNI
static void packB_int3_panel(const uint64_t* Bw, int wordsN, int k0, int kb, int jw, int8_t* Bp) {
	int kb4 = (kb + 3) & ~3;
	memset(Bp, 0, (size_t)(kb4 / 4) * 64);
	alignas(16) uint8_t row[4][16];
	int k;
	for (k = 0; k + 4 <= kb; k += 4) {
		for (int t = 0; t < 4; ++t) {
			uint64_t w = (Bw[(size_t)(k0 + k + t) * (size_t)wordsN + (size_t)jw] & int3_kValueMask) ^
			             0x4444444444444444ULL;
			__m128i b = _mm_cvtsi64_si128((long long)w);
			__m128i lo = _mm_and_si128(b, _mm_set1_epi8(0x0f));
			__m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(0x0f));
			_mm_store_si128((__m128i*)row[t], _mm_unpacklo_epi8(lo, hi));
		}
		int8_t* out = Bp + ((size_t)k / 4) * 64;
		__m128i v0 = _mm_load_si128((const __m128i*)row[0]);
		__m128i v1 = _mm_load_si128((const __m128i*)row[1]);
		__m128i v2 = _mm_load_si128((const __m128i*)row[2]);
		__m128i v3 = _mm_load_si128((const __m128i*)row[3]);
		__m128i a = _mm_unpacklo_epi8(v0, v1);
		__m128i b = _mm_unpackhi_epi8(v0, v1);
		__m128i c = _mm_unpacklo_epi8(v2, v3);
		__m128i d = _mm_unpackhi_epi8(v2, v3);
		_mm_storeu_si128((__m128i*)(out + 0), _mm_unpacklo_epi16(a, c));
		_mm_storeu_si128((__m128i*)(out + 16), _mm_unpackhi_epi16(a, c));
		_mm_storeu_si128((__m128i*)(out + 32), _mm_unpacklo_epi16(b, d));
		_mm_storeu_si128((__m128i*)(out + 48), _mm_unpackhi_epi16(b, d));
	}
	if (k < kb) {
		int8_t* out = Bp + ((size_t)k / 4) * 64;
		for (int t = 0; k + t < kb; ++t) {
			uint64_t w = (Bw[(size_t)(k0 + k + t) * (size_t)wordsN + (size_t)jw] & int3_kValueMask) ^
			             0x4444444444444444ULL;
			for (int j = 0; j < 16; ++j)
				out[j * 4 + t] = (int8_t)((w >> (unsigned)(j * 4)) & 7u);
		}
	}
}

struct I3PackedJob {
	const uint8_t* A;
	const uint64_t* Bw;
	int32_t* C;
	const int32_t* row;
	const int32_t* col;
	int ldc, M, N, K;
};

static void gemmINT3_fromPackedB_strip(void* ctx, int j0, int j1) {
	const I3PackedJob* job = (const I3PackedJob*)ctx;
	const uint8_t* A = job->A;
	const uint64_t* Bw = job->Bw;
	int32_t* C = job->C;
	int ldc = job->ldc;
	int M = job->M;
	int N = job->N;
	int K = job->K;
	int wordsN = N / 16;
	int nb = j1 - j0;
	if (nb <= 0)
		return;
	int kcStep = (K > 1024) ? kI8KC_HI : kI8KC_LO;
	if (kcStep > K)
		kcStep = K;
	int nPanels = (nb + kI8NR - 1) / kI8NR;
	size_t panelBytes = ((size_t)((kcStep + 3) / 4)) * 64;
	int8_t* Bp = (int8_t*)alloc64((size_t)nPanels * panelBytes);
	uint8_t* Ap = (uint8_t*)alloc64((size_t)kI8MC * (size_t)kcStep);
	for (int kc = 0; kc < K; kc += kcStep) {
		int kb = K - kc;
		if (kb > kcStep)
			kb = kcStep;
		for (int p = 0; p < nPanels; ++p)
			packB_int3_panel(Bw, wordsN, kc, kb, (j0 / 16) + p, Bp + (size_t)p * panelBytes);
		for (int ic = 0; ic < M; ic += kI8MC) {
			int mb = M - ic;
			if (mb > kI8MC)
				mb = kI8MC;
			for (int i = 0; i < mb; ++i)
				memcpy(Ap + (size_t)i * (size_t)kb,
				       A + (size_t)(ic + i) * (size_t)K + (size_t)kc, (size_t)kb);
			for (int ir = 0; ir < mb; ir += kI8MR) {
				int mr = mb - ir;
				if (mr > kI8MR)
					mr = kI8MR;
				for (int p = 0; p < nPanels; ++p) {
					int jr = p * kI8NR;
					int nr = nb - jr;
					if (nr > kI8NR)
						nr = kI8NR;
					kernelI8_16x16(Ap + (size_t)ir * (size_t)kb, kb,
					               Bp + (size_t)p * panelBytes,
					               C + (size_t)(ic + ir) * (size_t)ldc + (size_t)(j0 + jr),
					               ldc, kb, mr, nr);
				}
			}
		}
	}
	free64(Ap);
	free64(Bp);
	for (int i = 0; i < M; ++i) {
		int32_t rs = job->row[i];
		int32_t* crow = C + (size_t)i * (size_t)ldc;
		for (int j = j0; j < j1; ++j)
			crow[j] = crow[j] - 4 * rs - 4 * job->col[j] + 16 * K;
	}
}
#endif

static void gemmINT3(const uint64_t* Aw, const uint64_t* Bw, int32_t* C, int ldc,
                     int M, int N, int K, bool wrap) {
	if (wrap) {
		memoryAdviseWillNeed((void*)Aw, int3_bufferBytes((size_t)M * (size_t)K));
		memoryAdviseWillNeed((void*)Bw, int3_bufferBytes((size_t)K * (size_t)N));
	}

	if (N == 1) {
		int32_t* x = (int32_t*)alloc64((size_t)K * sizeof(int32_t));
		unpackInt3SignedI32(x, Bw, (size_t)K);
		if (((size_t)K % 16u) == 0)
			gemvINT3_streamA(Aw, x, C, M, K);
		else {
			for (int i = 0; i < M; ++i) {
				int32_t acc = 0;
				for (int k = 0; k < K; ++k)
					acc += int3_getSigned(Aw, (size_t)i * (size_t)K + (size_t)k) * x[k];
				C[i * ldc] = acc;
			}
		}
		free64(x);
		return;
	}

	if ((N % 16) == 0 && K >= 16 &&
	    ((size_t)K * (size_t)N >= 262144ull) &&
	    (M <= 64 || (size_t)N >= 4ull * (size_t)M)) {
		int32_t* X = (int32_t*)alloc64((size_t)M * (size_t)K * sizeof(int32_t));
		unpackInt3SignedI32(X, Aw, (size_t)M * (size_t)K);
		gemmINT3_streamB(X, Bw, C, ldc, M, N, K);
		free64(X);
		return;
	}

#if NDM_VNNI
	if ((N % 16) == 0 && K >= 16 && M >= 32 &&
	    (size_t)K * (size_t)N >= 1048576ull) {
		uint8_t* A = (uint8_t*)alloc64((size_t)M * (size_t)K);
		unpackInt3Xor4(A, Aw, (size_t)M * (size_t)K);
		int32_t* row = (int32_t*)calloc((size_t)M, sizeof(int32_t));
		int32_t* col = (int32_t*)calloc((size_t)N, sizeof(int32_t));
		if (!row || !col) {
			free(row);
			free(col);
			free64(A);
			throw std::bad_alloc();
		}
		for (int i = 0; i < M; ++i) {
			int32_t s = 0;
			for (int k = 0; k < K; ++k)
				s += (int32_t)A[(size_t)i * (size_t)K + (size_t)k];
			row[i] = s;
		}
		int wordsN = N / 16;
		for (int k = 0; k < K; ++k) {
			for (int jw = 0; jw < wordsN; ++jw) {
				uint64_t r = (Bw[(size_t)k * (size_t)wordsN + (size_t)jw] & int3_kValueMask) ^
				             0x4444444444444444ULL;
				int32_t* cp = col + jw * 16;
				for (int j = 0; j < 16; ++j)
					cp[j] += (int32_t)((r >> (unsigned)(j * 4)) & 7u);
			}
		}
		memset(C, 0, (size_t)M * (size_t)ldc * sizeof(int32_t));
		I3PackedJob job;
		job.A = A;
		job.Bw = Bw;
		job.C = C;
		job.row = row;
		job.col = col;
		job.ldc = ldc;
		job.M = M;
		job.N = N;
		job.K = K;
		ndmRunNStrips(N, 16, (size_t)M * (size_t)N * (size_t)K, gemmINT3_fromPackedB_strip, &job);
		free(row);
		free(col);
		free64(A);
		return;
	}
#endif

	uint8_t* A = (uint8_t*)alloc64((size_t)M * (size_t)K);
	uint8_t* B = (uint8_t*)alloc64((size_t)K * (size_t)N);
	unpackInt3Xor4(A, Aw, (size_t)M * (size_t)K);
	unpackInt3Xor4(B, Bw, (size_t)K * (size_t)N);
	memset(C, 0, (size_t)M * (size_t)ldc * sizeof(int32_t));
#if NDM_VNNI
	int32_t* row = (int32_t*)calloc((size_t)M, sizeof(int32_t));
	int32_t* col = (int32_t*)calloc((size_t)N, sizeof(int32_t));
	if (!row || !col) {
		free(row);
		free(col);
		free64(A);
		free64(B);
		throw std::bad_alloc();
	}
	for (int i = 0; i < M; ++i) {
		int32_t s = 0;
		for (int k = 0; k < K; ++k)
			s += (int32_t)A[(size_t)i * (size_t)K + (size_t)k];
		row[i] = s;
	}
	for (int j = 0; j < N; ++j) {
		int32_t s = 0;
		for (int k = 0; k < K; ++k)
			s += (int32_t)B[(size_t)k * (size_t)N + (size_t)j];
		col[j] = s;
	}
	gemmVNNI_u8i8(A, K, (const int8_t*)B, N, C, ldc, M, N, K, 0, 0);
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
			C[i * ldc + j] = C[i * ldc + j] - 4 * row[i] - 4 * col[j] + 16 * K;
	free(row);
	free(col);
#else
	int8_t* As = (int8_t*)A;
	int8_t* Bs = (int8_t*)B;
	for (int t = 0; t < M * K; ++t)
		As[t] = (int8_t)((int)A[t] - 4);
	for (int t = 0; t < K * N; ++t)
		Bs[t] = (int8_t)((int)((uint8_t*)B)[t] - 4);
	gemmI8_scalar(As, K, Bs, N, C, ldc, M, N, K);
#endif
	free64(A);
	free64(B);
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

static void gemmBINARY(const uint64_t* Aw, const uint64_t* Bw, int32_t* C, int ldc,
                       int M, int N, int K, bool wrap) {
	memset(C, 0, (size_t)M * (size_t)ldc * sizeof(int32_t));
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
	if (N == 1)
		packBitRow(bCols, Bw, 0, K);
	else if ((K % 64) == 0 && (N % 64) == 0) {
		int nWords = N / 64;
		for (int jw = 0; jw < nWords; ++jw) {
			for (int kw = 0; kw < kWords; ++kw) {
				uint64_t tile[64];
				uint64_t out[64];
				for (int r = 0; r < 64; ++r) {
					size_t bit = ((size_t)kw * 64 + (size_t)r) * (size_t)N + (size_t)jw * 64;
					tile[r] = Bw[bit >> 6];
					out[r] = 0;
				}
				for (int r = 0; r < 64; ++r) {
					uint64_t x = tile[r];
					for (int c = 0; c < 64; ++c)
						out[c] |= ((x >> c) & 1ULL) << r;
				}
				for (int c = 0; c < 64; ++c)
					bCols[(size_t)(jw * 64 + c) * (size_t)kWords + (size_t)kw] = out[c];
			}
		}
	} else {
		for (int j = 0; j < N; ++j) {
			uint64_t* col = bCols + (size_t)j * (size_t)kWords;
			for (int k = 0; k < K; ++k) {
				size_t bit = (size_t)k * (size_t)N + (size_t)j;
				if ((Bw[bit >> 6] >> (bit & 63u)) & 1u)
					col[k >> 6] |= (uint64_t)1 << (k & 63);
			}
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
			C[i * ldc + j] = (int32_t)acc;
		}
	}
	free(aRows);
	free(bCols);
}

static void gemmINT32(const int32_t* A, int lda, const int32_t* B, int ldb,
                      int64_t* C, int ldc, int M, int N, int K, bool wrapA, bool wrapB) {
	memset(C, 0, (size_t)M * (size_t)ldc * sizeof(int64_t));
	int32_t* Bp = (int32_t*)alloc64((size_t)kKC * (size_t)kNR * sizeof(int32_t));
	int32_t* Ap = (int32_t*)alloc64((size_t)kMC * (size_t)kKC * sizeof(int32_t));
	for (int kc = 0; kc < K; kc += kKC) {
		int kb = K - kc;
		if (kb > kKC)
			kb = kKC;
		advisePanel(B, 4, ldb, kc, kb, 0, N, wrapB);
		for (int jc = 0; jc < N; jc += kNR) {
			int nb = N - jc;
			if (nb > kNR)
				nb = kNR;
			for (int k = 0; k < kb; ++k) {
				memcpy(Bp + (size_t)k * (size_t)kNR,
				       B + (size_t)(kc + k) * (size_t)ldb + (size_t)jc, (size_t)nb * sizeof(int32_t));
				if (nb < kNR)
					memset(Bp + (size_t)k * (size_t)kNR + nb, 0, (size_t)(kNR - nb) * sizeof(int32_t));
			}
			for (int ic = 0; ic < M; ic += kMC) {
				int mb = M - ic;
				if (mb > kMC)
					mb = kMC;
				advisePanel(A, 4, lda, ic, mb, kc, kb, wrapA);
				for (int i = 0; i < mb; ++i)
					memcpy(Ap + (size_t)i * (size_t)kb,
					       A + (size_t)(ic + i) * (size_t)lda + (size_t)kc, (size_t)kb * sizeof(int32_t));
				for (int ir = 0; ir < mb; ++ir) {
					int64_t* crow = C + (size_t)(ic + ir) * (size_t)ldc + (size_t)jc;
					const int32_t* arow = Ap + (size_t)ir * (size_t)kb;
#if NDM_AVX512DQ
					if (nb == 16) {
						__m512i acc_lo = _mm512_loadu_si512(crow);
						__m512i acc_hi = _mm512_loadu_si512(crow + 8);
						for (int k = 0; k < kb; ++k) {
							__m512i av = _mm512_set1_epi64((int64_t)arow[k]);
							__m512i bv = _mm512_loadu_si512((const void*)(Bp + (size_t)k * (size_t)kNR));
							__m512i b64a = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(bv));
							__m512i b64b = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(bv, 1));
							acc_lo = _mm512_add_epi64(acc_lo, _mm512_mullo_epi64(av, b64a));
							acc_hi = _mm512_add_epi64(acc_hi, _mm512_mullo_epi64(av, b64b));
						}
						_mm512_storeu_si512(crow, acc_lo);
						_mm512_storeu_si512(crow + 8, acc_hi);
						continue;
					}
#endif
					for (int j = 0; j < nb; ++j) {
						int64_t acc = crow[j];
						for (int k = 0; k < kb; ++k)
							acc += (int64_t)arow[k] * (int64_t)Bp[(size_t)k * (size_t)kNR + (size_t)j];
						crow[j] = acc;
					}
				}
			}
		}
	}
	free64(Ap);
	free64(Bp);
}

static void gemmINT64(const int64_t* A, int lda, const int64_t* B, int ldb,
                      int64_t* C, int ldc, int M, int N, int K, bool wrapA, bool wrapB) {
	memset(C, 0, (size_t)M * (size_t)ldc * sizeof(int64_t));
	int64_t* Bp = (int64_t*)alloc64((size_t)kKC * (size_t)kNR * sizeof(int64_t));
	int64_t* Ap = (int64_t*)alloc64((size_t)kMC * (size_t)kKC * sizeof(int64_t));
	for (int kc = 0; kc < K; kc += kKC) {
		int kb = K - kc;
		if (kb > kKC)
			kb = kKC;
		advisePanel(B, 8, ldb, kc, kb, 0, N, wrapB);
		for (int jc = 0; jc < N; jc += kNR) {
			int nb = N - jc;
			if (nb > kNR)
				nb = kNR;
			for (int k = 0; k < kb; ++k) {
				memcpy(Bp + (size_t)k * (size_t)kNR,
				       B + (size_t)(kc + k) * (size_t)ldb + (size_t)jc, (size_t)nb * sizeof(int64_t));
				if (nb < kNR)
					memset(Bp + (size_t)k * (size_t)kNR + nb, 0, (size_t)(kNR - nb) * sizeof(int64_t));
			}
			for (int ic = 0; ic < M; ic += kMC) {
				int mb = M - ic;
				if (mb > kMC)
					mb = kMC;
				advisePanel(A, 8, lda, ic, mb, kc, kb, wrapA);
				for (int i = 0; i < mb; ++i)
					memcpy(Ap + (size_t)i * (size_t)kb,
					       A + (size_t)(ic + i) * (size_t)lda + (size_t)kc, (size_t)kb * sizeof(int64_t));
				for (int ir = 0; ir < mb; ++ir) {
					int64_t* crow = C + (size_t)(ic + ir) * (size_t)ldc + (size_t)jc;
					const int64_t* arow = Ap + (size_t)ir * (size_t)kb;
					for (int j = 0; j < nb; ++j) {
#if defined(__SIZEOF_INT128__)
						__int128 acc = (__int128)crow[j];
						for (int k = 0; k < kb; ++k)
							acc += (__int128)arow[k] * (__int128)Bp[(size_t)k * (size_t)kNR + (size_t)j];
						crow[j] = (int64_t)acc;
#else
						int64_t acc = crow[j];
						for (int k = 0; k < kb; ++k)
							acc += arow[k] * Bp[(size_t)k * (size_t)kNR + (size_t)j];
						crow[j] = acc;
#endif
					}
				}
			}
		}
	}
	free64(Ap);
	free64(Bp);
}

static void gemmU256(const uint256_t* A, int lda, const uint256_t* B, int ldb,
                     uint256_t* C, int ldc, int M, int N, int K) {
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			uint256_t acc(0);
			for (int k = 0; k < K; ++k)
				acc = acc + A[(size_t)i * (size_t)lda + (size_t)k] *
				            B[(size_t)k * (size_t)ldb + (size_t)j];
			C[(size_t)i * (size_t)ldc + (size_t)j] = acc;
		}
	}
}

template <typename Lane, typename Acc, typename Out>
static void gemvDense(const Lane* A, int lda, const Lane* x, Out* y, int M, int K) {
	for (int i = 0; i < M; ++i) {
		const Lane* arow = A + (size_t)i * (size_t)lda;
#if NDM_AVX512
		if (sizeof(Lane) == 4 && sizeof(Out) == 4) {
			__m512 vacc = _mm512_setzero_ps();
			int k = 0;
			for (; k + 16 <= K; k += 16)
				vacc = _mm512_fmadd_ps(_mm512_loadu_ps((const float*)(arow + k)),
				                       _mm512_loadu_ps((const float*)(x + k)), vacc);
			float s = _mm512_reduce_add_ps(vacc);
			for (; k < K; ++k)
				s += (float)arow[k] * (float)x[k];
			y[i] = (Out)s;
			continue;
		}
#endif
#if NDM_AVX2
		if (sizeof(Lane) == 4 && sizeof(Out) == 4) {
			__m256 vacc = _mm256_setzero_ps();
			int k = 0;
			for (; k + 8 <= K; k += 8)
				vacc = ndmFmaddPs(_mm256_loadu_ps((const float*)(arow + k)),
				                  _mm256_loadu_ps((const float*)(x + k)), vacc);
			alignas(32) float tmp[8];
			_mm256_store_ps(tmp, vacc);
			float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
			for (; k < K; ++k)
				s += (float)arow[k] * (float)x[k];
			y[i] = (Out)s;
			continue;
		}
#endif
#if defined(__SIZEOF_INT128__)
		if (sizeof(Lane) == 8 && sizeof(Acc) == 8 && sizeof(Out) == 8) {
			__int128 acc128 = 0;
			for (int k = 0; k < K; ++k)
				acc128 += (__int128)arow[k] * (__int128)x[k];
			y[i] = (Out)(int64_t)acc128;
			continue;
		}
#endif
		Acc acc = 0;
		for (int k = 0; k < K; ++k)
			acc += (Acc)arow[k] * (Acc)x[k];
		y[i] = (Out)acc;
	}
}


static NDArray dense2d(const NDArrayView& v, int rows, int cols, int batch, int b) {
	if (batch <= 1 && isPackedContig2d(v, rows, cols))
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
			else
				e += (size_t)b * st.get(0) + (size_t)i * st.get(1) + (size_t)j * st.get(2);
			switch (v.getType()) {
				case F16:
				case BF16:
					out.set({i, j}, ndarray_half::load(v.getType(), ((const uint16_t*)v.data())[e]));
					break;
				case F32: out.set({i, j}, ((const float*)v.data())[e]); break;
				case F64: out.set({i, j}, ((const double*)v.data())[e]); break;
				case UINT256: out.set({i, j}, ((const uint256_t*)v.data())[e]); break;
				case INT64: out.set({i, j}, ((const int64_t*)v.data())[e]); break;
				case INT32: out.set({i, j}, ((const int32_t*)v.data())[e]); break;
				case INT8: out.set({i, j}, (int)((const int8_t*)v.data())[e]); break;
				case UINT8: out.set({i, j}, (int)((const uint8_t*)v.data())[e]); break;
				case INT3: out.set({i, j}, int3_getSigned((const uint64_t*)v.data(), e)); break;
				case BINARY:
					out.set({i, j}, (int)((((const uint64_t*)v.data())[e >> 6] >> (e & 63u)) & 1u));
					break;
				default: break;
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
					if (batch > 1) c.set({b, i, j}, plane.get<float>({i, j}));
					else c.set({i, j}, plane.get<float>({i, j}));
					break;
				case F64:
					if (batch > 1) c.set({b, i, j}, plane.get<double>({i, j}));
					else c.set({i, j}, plane.get<double>({i, j}));
					break;
				case UINT256:
					if (batch > 1) c.set({b, i, j}, plane.get<uint256_t>({i, j}));
					else c.set({i, j}, plane.get<uint256_t>({i, j}));
					break;
				case INT64:
					if (batch > 1) c.set({b, i, j}, plane.get<int64_t>({i, j}));
					else c.set({i, j}, plane.get<int64_t>({i, j}));
					break;
				default:
					if (batch > 1) c.set({b, i, j}, plane.get<int>({i, j}));
					else c.set({i, j}, plane.get<int>({i, j}));
					break;
			}
		}
	}
}

static NDArray gemmRaw(NDArrayType type, const void* ap, int lda, const void* bp, int ldb,
                       int M, int N, int K, bool wrapA, bool wrapB) {
	NDArray c({M, N}, matmulResultType(type));
	void* cp = c.data();
	if (M == 0 || N == 0)
		return c;
	if (!ap || !bp)
		throw std::invalid_argument("NDArray::matmul - null buffer");

	if (N == 1) {
		switch (type) {
			case F16:
				gemvHalf(NdmHalf::F16, (const uint16_t*)ap, lda, (const uint16_t*)bp, (float*)cp, M, K);
				return c;
			case BF16:
				gemvHalf(NdmHalf::BF16, (const uint16_t*)ap, lda, (const uint16_t*)bp, (float*)cp, M, K);
				return c;
			case F32:
				gemvDense<float, float, float>((const float*)ap, lda, (const float*)bp, (float*)cp, M, K);
				return c;
			case F64:
				gemvDense<double, double, double>((const double*)ap, lda, (const double*)bp, (double*)cp, M, K);
				return c;
			case INT8:
				gemvDense<int8_t, int32_t, int32_t>((const int8_t*)ap, lda, (const int8_t*)bp, (int32_t*)cp, M, K);
				return c;
			case UINT8:
				gemvDense<uint8_t, int32_t, int32_t>((const uint8_t*)ap, lda, (const uint8_t*)bp, (int32_t*)cp, M, K);
				return c;
			case INT32:
				gemvDense<int32_t, int64_t, int64_t>((const int32_t*)ap, lda, (const int32_t*)bp, (int64_t*)cp, M, K);
				return c;
			case INT64:
				gemvDense<int64_t, int64_t, int64_t>((const int64_t*)ap, lda, (const int64_t*)bp, (int64_t*)cp, M, K);
				return c;
			default:
				break;
		}
	}

	switch (type) {
		case F16:
			gemmHalf(NdmHalf::F16, (const uint16_t*)ap, lda, (const uint16_t*)bp, ldb,
			         (float*)cp, N, M, N, K, wrapA, wrapB);
			break;
		case BF16:
			gemmHalf(NdmHalf::BF16, (const uint16_t*)ap, lda, (const uint16_t*)bp, ldb,
			         (float*)cp, N, M, N, K, wrapA, wrapB);
			break;
		case F32:
			gemmF32((const float*)ap, lda, (const float*)bp, ldb, (float*)cp, N, M, N, K, wrapA, wrapB);
			break;
		case F64:
			gemmF64((const double*)ap, lda, (const double*)bp, ldb, (double*)cp, N, M, N, K, wrapA, wrapB);
			break;
		case INT8:
			gemmINT8((const int8_t*)ap, lda, (const int8_t*)bp, ldb, (int32_t*)cp, N, M, N, K);
			break;
		case UINT8:
			gemmUINT8((const uint8_t*)ap, lda, (const uint8_t*)bp, ldb, (int32_t*)cp, N, M, N, K);
			break;
		case INT32:
			gemmINT32((const int32_t*)ap, lda, (const int32_t*)bp, ldb, (int64_t*)cp, N, M, N, K, wrapA, wrapB);
			break;
		case INT64:
			gemmINT64((const int64_t*)ap, lda, (const int64_t*)bp, ldb, (int64_t*)cp, N, M, N, K, wrapA, wrapB);
			break;
		case UINT256:
			gemmU256((const uint256_t*)ap, lda, (const uint256_t*)bp, ldb, (uint256_t*)cp, N, M, N, K);
			break;
		case INT3:
			if (lda == K && ldb == N)
				gemmINT3((const uint64_t*)ap, (const uint64_t*)bp, (int32_t*)cp, N, M, N, K, wrapA || wrapB);
			else {
				NDArray ad({M, K}, INT3);
				NDArray bd({K, N}, INT3);
				const uint64_t* As = (const uint64_t*)ap;
				const uint64_t* Bs = (const uint64_t*)bp;
				for (int i = 0; i < M; ++i)
					for (int k = 0; k < K; ++k)
						ad.set({i, k}, int3_getSigned(As, (size_t)i * (size_t)lda + (size_t)k));
				for (int k = 0; k < K; ++k)
					for (int j = 0; j < N; ++j)
						bd.set({k, j}, int3_getSigned(Bs, (size_t)k * (size_t)ldb + (size_t)j));
				return gemmRaw(INT3, ad.data(), K, bd.data(), N, M, N, K, false, false);
			}
			break;
		case BINARY:
			if (lda == K && ldb == N)
				gemmBINARY((const uint64_t*)ap, (const uint64_t*)bp, (int32_t*)cp, N, M, N, K, wrapA || wrapB);
			else {
				NDArray ad({M, K}, BINARY);
				NDArray bd({K, N}, BINARY);
				const uint64_t* As = (const uint64_t*)ap;
				const uint64_t* Bs = (const uint64_t*)bp;
				for (int i = 0; i < M; ++i)
					for (int k = 0; k < K; ++k) {
						size_t e = (size_t)i * (size_t)lda + (size_t)k;
						ad.set({i, k}, (int)((As[e >> 6] >> (e & 63u)) & 1u));
					}
				for (int k = 0; k < K; ++k)
					for (int j = 0; j < N; ++j) {
						size_t e = (size_t)k * (size_t)ldb + (size_t)j;
						bd.set({k, j}, (int)((Bs[e >> 6] >> (e & 63u)) & 1u));
					}
				return gemmRaw(BINARY, ad.data(), K, bd.data(), N, M, N, K, false, false);
			}
			break;
		default:
			throw std::invalid_argument("NDArray::matmul - unsupported type");
	}
	return c;
}

static NDArray gemmViews2d(const NDArrayView& a, const NDArrayView& b, int M, int N, int K,
                           bool wrapA, bool wrapB) {
	if (isPackedContig2d(a, M, K) && isPackedContig2d(b, K, N)) {
		const void* ap = contigElemPtr(a);
		const void* bp = contigElemPtr(b);
		if (ap && bp)
			return gemmRaw(a.getType(), ap, K, bp, N, M, N, K, wrapA, wrapB);
	}
	NDArray ad = dense2d(a, M, K, 1, 0);
	NDArray bd = dense2d(b, K, N, 1, 0);
	return gemmRaw(a.getType(), ad.data(), K, bd.data(), N, M, N, K, false, false);
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

static void last2(const ArrayList<int>& sh, int& x, int& y) {
	int r = sh.size();
	x = sh.get(r - 2);
	y = sh.get(r - 1);
}

static MatmulGeom parseGeom(const NDArrayView& a, const NDArrayView& b) {
	int ar = a.getShape().size();
	int br = b.getShape().size();
	MatmulGeom g;

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

	if (g.batch == 1 && !g.squeezeM && !g.squeezeN)
		return gemmViews2d(a, b, g.M, g.N, g.K, wrapA, wrapB);

	if (g.batch == 1 && (g.squeezeM || g.squeezeN)) {
		NDArray plane = gemmViews2d(a, b, g.M, g.N, g.K, wrapA, wrapB);
		if (g.squeezeM && g.squeezeN) {
			NDArray acc(outShape, plane.type);
			switch (plane.type) {
				case F32: acc.set({}, plane.get<float>({0, 0})); break;
				case F64: acc.set({}, plane.get<double>({0, 0})); break;
				case UINT256: acc.set({}, plane.get<uint256_t>({0, 0})); break;
				case INT64: acc.set({}, plane.get<int64_t>({0, 0})); break;
				default: acc.set({}, plane.get<int>({0, 0})); break;
			}
			return acc;
		}
		NDArray acc(outShape, plane.type);
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
		return acc;
	}

	NDArray acc(outShape, matmulResultType(a.getType()));
	const size_t es = denseElemSize(a.getType());
	const bool aPlane = (es != 0) && a.data() &&
		((g.aHasBatch && a.isContiguous() && a.getShape().size() == 3) ||
		 (!g.aHasBatch && isPackedContig2d(a, g.M, g.K)));
	const bool bPlane = (es != 0) && b.data() &&
		((g.bHasBatch && b.isContiguous() && b.getShape().size() == 3) ||
		 (!g.bHasBatch && isPackedContig2d(b, g.K, g.N)));
	for (int bi = 0; bi < g.batch; ++bi) {
		int ab = g.aHasBatch ? bi : 0;
		int bb = g.bHasBatch ? bi : 0;
		int aBatchDim = g.aHasBatch ? g.batch : 1;
		int bBatchDim = g.bHasBatch ? g.batch : 1;
		NDArray ad;
		NDArray bd;
		const void* ap;
		const void* bp;
		if (aPlane) {
			ap = (const char*)a.data() +
			     (a.getOffset() + (size_t)ab * (size_t)g.M * (size_t)g.K) * es;
		} else {
			ad = dense2d(a, g.M, g.K, aBatchDim, ab);
			ap = ad.data();
		}
		if (bPlane) {
			bp = (const char*)b.data() +
			     (b.getOffset() + (size_t)bb * (size_t)g.K * (size_t)g.N) * es;
		} else {
			bd = dense2d(b, g.K, g.N, bBatchDim, bb);
			bp = bd.data();
		}
		NDArray plane = gemmRaw(a.getType(), ap, g.K, bp, g.N, g.M, g.N, g.K, false, false);
		storePlane(acc, plane, g.batch, bi);
	}
	return acc;
}
