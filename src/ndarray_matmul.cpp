// Tiled matmul / gemv. Algorithms adapted from laptopnn INT3/BINARY/INT8 GEMM
// (BLIS pack + ISA MAC), rewritten for every NDArrayType.

#include "ndarray_matmul.h"

#include "NDArray.h"
#include "fs/FdHandle.h"

#include <cstring>
#include <stdexcept>

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

#if defined(__AVX2__)
#define NDM_AVX2 1
#ifndef _MSC_VER
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

#if defined(__AVX512VNNI__) || defined(__AVXVNNI__)
#define NDM_VNNI 1
#else
#define NDM_VNNI 0
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

static int loadInt3(const uint64_t* w, size_t i) {
	uint8_t p = (uint8_t)((w[i / 16] >> (unsigned)((i % 16) * 4)) & 7u);
	return p >= 4 ? (int)p - 8 : (int)p;
}

static int loadBit(const uint64_t* w, size_t i) {
	return (int)((w[i >> 6] >> (i & 63u)) & 1u);
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

static void maybeAdvise(const void* p, size_t bytes, bool wrap) {
	if (wrap && p && bytes)
		memoryAdviseWillNeed((void*)p, bytes);
}


// ---- F32 / F64 tiled -------------------------------------------------------

template <typename T>
static void gemmFloat(const T* NDM_RESTRICT A, const T* NDM_RESTRICT B, T* NDM_RESTRICT C, int M, int N, int K, bool wrapA, bool wrapB) {
	memset(C, 0, (size_t)M * (size_t)N * sizeof(T));
	T* Bp = (T*)malloc((size_t)kKC * (size_t)kNR * sizeof(T));
	T* Ap = (T*)malloc((size_t)kMC * (size_t)kKC * sizeof(T));
	if (!Bp || !Ap) {
		free(Bp);
		free(Ap);
		throw std::bad_alloc();
	}

	for (int kc = 0; kc < K; kc += kKC) {
		int kb = K - kc;
		if (kb > kKC)
			kb = kKC;
		if (wrapB)
			maybeAdvise(B + (size_t)kc * (size_t)N, (size_t)kb * (size_t)N * sizeof(T), true);

		for (int jc = 0; jc < N; jc += kNR) {
			int nb = N - jc;
			if (nb > kNR)
				nb = kNR;
			for (int k = 0; k < kb; ++k)
				memcpy(Bp + (size_t)k * (size_t)nb,
				       B + (size_t)(kc + k) * (size_t)N + (size_t)jc,
				       (size_t)nb * sizeof(T));

			for (int ic = 0; ic < M; ic += kMC) {
				int mb = M - ic;
				if (mb > kMC)
					mb = kMC;
				if (wrapA && jc == 0)
					maybeAdvise(A + (size_t)ic * (size_t)K + (size_t)kc,
					            (size_t)mb * (size_t)K * sizeof(T), true);
				for (int i = 0; i < mb; ++i)
					memcpy(Ap + (size_t)i * (size_t)kb,
					       A + (size_t)(ic + i) * (size_t)K + (size_t)kc,
					       (size_t)kb * sizeof(T));

				for (int ir = 0; ir < mb; ++ir) {
					T* crow = C + (size_t)(ic + ir) * (size_t)N + (size_t)jc;
					const T* arow = Ap + (size_t)ir * (size_t)kb;
#if NDM_AVX2 && NDM_FMA
					if (sizeof(T) == 4 && nb == 8) {
						__m256 acc = _mm256_loadu_ps((const float*)crow);
						for (int k = 0; k < kb; ++k) {
							__m256 av = _mm256_broadcast_ss((const float*)(arow + k));
							__m256 bv = _mm256_loadu_ps((const float*)(Bp + (size_t)k * 8));
							acc = _mm256_fmadd_ps(av, bv, acc);
						}
						_mm256_storeu_ps((float*)crow, acc);
						continue;
					}
#endif
#if NDM_NEON
					if (sizeof(T) == 4 && nb == 8) {
						float32x4_t acc0 = vld1q_f32((const float*)crow);
						float32x4_t acc1 = vld1q_f32((const float*)crow + 4);
						for (int k = 0; k < kb; ++k) {
							float32x4_t av = vdupq_n_f32((float)arow[k]);
							const float* bp = (const float*)(Bp + (size_t)k * 8);
							acc0 = vfmaq_f32(acc0, av, vld1q_f32(bp));
							acc1 = vfmaq_f32(acc1, av, vld1q_f32(bp + 4));
						}
						vst1q_f32((float*)crow, acc0);
						vst1q_f32((float*)crow + 4, acc1);
						continue;
					}
#endif
					for (int j = 0; j < nb; ++j) {
						T acc = crow[j];
						for (int k = 0; k < kb; ++k)
							acc += arow[k] * Bp[(size_t)k * (size_t)nb + (size_t)j];
						crow[j] = acc;
					}
				}
			}
		}
	}
	free(Ap);
	free(Bp);
}


// ---- integer tiled (Lane -> Acc -> Out) ------------------------------------

template <typename Lane, typename Acc, typename Out>
static void gemmInt(const Lane* NDM_RESTRICT A, const Lane* NDM_RESTRICT B, Out* NDM_RESTRICT C,
                    int M, int N, int K, bool wrapA, bool wrapB) {
	memset(C, 0, (size_t)M * (size_t)N * sizeof(Out));
	Lane* Bp = (Lane*)malloc((size_t)kKC * (size_t)kNR * sizeof(Lane));
	Lane* Ap = (Lane*)malloc((size_t)kMC * (size_t)kKC * sizeof(Lane));
	if (!Bp || !Ap) {
		free(Bp);
		free(Ap);
		throw std::bad_alloc();
	}

	for (int kc = 0; kc < K; kc += kKC) {
		int kb = K - kc;
		if (kb > kKC)
			kb = kKC;
		if (wrapB)
			maybeAdvise(B + (size_t)kc * (size_t)N, (size_t)kb * (size_t)N * sizeof(Lane), true);

		for (int jc = 0; jc < N; jc += kNR) {
			int nb = N - jc;
			if (nb > kNR)
				nb = kNR;
			for (int k = 0; k < kb; ++k)
				memcpy(Bp + (size_t)k * (size_t)nb,
				       B + (size_t)(kc + k) * (size_t)N + (size_t)jc,
				       (size_t)nb * sizeof(Lane));

			for (int ic = 0; ic < M; ic += kMC) {
				int mb = M - ic;
				if (mb > kMC)
					mb = kMC;
				for (int i = 0; i < mb; ++i)
					memcpy(Ap + (size_t)i * (size_t)kb,
					       A + (size_t)(ic + i) * (size_t)K + (size_t)kc,
					       (size_t)kb * sizeof(Lane));

				for (int ir = 0; ir < mb; ++ir) {
					Out* crow = C + (size_t)(ic + ir) * (size_t)N + (size_t)jc;
					const Lane* arow = Ap + (size_t)ir * (size_t)kb;
					for (int j = 0; j < nb; ++j) {
						Acc acc = (Acc)crow[j];
						for (int k = 0; k < kb; ++k)
							acc += (Acc)arow[k] * (Acc)Bp[(size_t)k * (size_t)nb + (size_t)j];
						crow[j] = (Out)acc;
					}
				}
			}
		}
	}
	free(Ap);
	free(Bp);
}


static void gemmInt64(const int64_t* A, const int64_t* B, int64_t* C, int M, int N, int K,
                      bool wrapA, bool wrapB) {
	memset(C, 0, (size_t)M * (size_t)N * sizeof(int64_t));
	int64_t* Bp = (int64_t*)malloc((size_t)kKC * (size_t)kNR * sizeof(int64_t));
	int64_t* Ap = (int64_t*)malloc((size_t)kMC * (size_t)kKC * sizeof(int64_t));
	if (!Bp || !Ap) {
		free(Bp);
		free(Ap);
		throw std::bad_alloc();
	}
	for (int kc = 0; kc < K; kc += kKC) {
		int kb = K - kc;
		if (kb > kKC)
			kb = kKC;
		if (wrapB)
			maybeAdvise(B + (size_t)kc * (size_t)N, (size_t)kb * (size_t)N * sizeof(int64_t), true);
		for (int jc = 0; jc < N; jc += kNR) {
			int nb = N - jc;
			if (nb > kNR)
				nb = kNR;
			for (int k = 0; k < kb; ++k)
				memcpy(Bp + (size_t)k * (size_t)nb,
				       B + (size_t)(kc + k) * (size_t)N + (size_t)jc,
				       (size_t)nb * sizeof(int64_t));
			for (int ic = 0; ic < M; ic += kMC) {
				int mb = M - ic;
				if (mb > kMC)
					mb = kMC;
				for (int i = 0; i < mb; ++i)
					memcpy(Ap + (size_t)i * (size_t)kb,
					       A + (size_t)(ic + i) * (size_t)K + (size_t)kc,
					       (size_t)kb * sizeof(int64_t));
				for (int ir = 0; ir < mb; ++ir) {
					int64_t* crow = C + (size_t)(ic + ir) * (size_t)N + (size_t)jc;
					const int64_t* arow = Ap + (size_t)ir * (size_t)kb;
					for (int j = 0; j < nb; ++j) {
#if defined(__SIZEOF_INT128__)
						__int128 acc = (__int128)crow[j];
						for (int k = 0; k < kb; ++k)
							acc += (__int128)arow[k] * (__int128)Bp[(size_t)k * (size_t)nb + (size_t)j];
						crow[j] = (int64_t)acc;
#else
						int64_t acc = crow[j];
						for (int k = 0; k < kb; ++k)
							acc += arow[k] * Bp[(size_t)k * (size_t)nb + (size_t)j];
						crow[j] = acc;
#endif
					}
				}
			}
		}
	}
	free(Ap);
	free(Bp);
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

static void gemmInt3(const uint64_t* Aw, const uint64_t* Bw, int32_t* C, int M, int N, int K,
                     bool wrap) {
	int8_t* A = (int8_t*)malloc((size_t)M * (size_t)K);
	int8_t* B = (int8_t*)malloc((size_t)K * (size_t)N);
	if (!A || !B) {
		free(A);
		free(B);
		throw std::bad_alloc();
	}
	maybeAdvise(Aw, ((size_t)M * (size_t)K + 15) / 16 * 8, wrap);
	maybeAdvise(Bw, ((size_t)K * (size_t)N + 15) / 16 * 8, wrap);
	for (int i = 0; i < M; ++i)
		for (int k = 0; k < K; ++k)
			A[(size_t)i * (size_t)K + (size_t)k] = (int8_t)loadInt3(Aw, (size_t)i * (size_t)K + (size_t)k);
	for (int k = 0; k < K; ++k)
		for (int j = 0; j < N; ++j)
			B[(size_t)k * (size_t)N + (size_t)j] = (int8_t)loadInt3(Bw, (size_t)k * (size_t)N + (size_t)j);
	gemmInt<int8_t, int32_t, int32_t>(A, B, C, M, N, K, false, false);
	free(A);
	free(B);
}

static void gemmBinary(const uint64_t* Aw, const uint64_t* Bw, int32_t* C, int M, int N, int K,
                       bool wrap) {
	memset(C, 0, (size_t)M * (size_t)N * sizeof(int32_t));
	maybeAdvise(Aw, ((size_t)M * (size_t)K + 63) / 64 * 8, wrap);
	maybeAdvise(Bw, ((size_t)K * (size_t)N + 63) / 64 * 8, wrap);

	int kWords = (K + 63) / 64;
	uint64_t* aRows = (uint64_t*)calloc((size_t)M * (size_t)kWords, sizeof(uint64_t));
	uint64_t* bCols = (uint64_t*)calloc((size_t)N * (size_t)kWords, sizeof(uint64_t));
	if (!aRows || !bCols) {
		free(aRows);
		free(bCols);
		throw std::bad_alloc();
	}
	for (int i = 0; i < M; ++i)
		for (int k = 0; k < K; ++k)
			if (loadBit(Aw, (size_t)i * (size_t)K + (size_t)k))
				aRows[(size_t)i * (size_t)kWords + (size_t)(k >> 6)] |= (uint64_t)1 << (k & 63);
	for (int j = 0; j < N; ++j)
		for (int k = 0; k < K; ++k)
			if (loadBit(Bw, (size_t)k * (size_t)N + (size_t)j))
				bCols[(size_t)j * (size_t)kWords + (size_t)(k >> 6)] |= (uint64_t)1 << (k & 63);

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


static bool viewIsWrap(const NDArrayView& v) {
	if (!v.sharedBuffer() || !v.sharedBuffer().get())
		return false;
	return !v.sharedBuffer().get()->ownsData;
}

static NDArray dense2d(const NDArrayView& v, int rows, int cols, int batch, int b) {
	NDArray out({rows, cols}, v.getType());
	if (batch <= 1 && v.isContiguous() && v.getShape().size() == 2)
		return v.copy();
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			ArrayList<int> idx;
			if (v.getShape().size() == 1) {
				idx.add(rows == 1 ? j : i);
			} else {
				idx.add(i);
				idx.add(j);
				if (batch > 1)
					idx.add(b);
			}
			switch (v.getType()) {
				case F32: out.set({i, j}, v.get<float>(idx)); break;
				case F64: out.set({i, j}, v.get<double>(idx)); break;
				case UINT256: out.set({i, j}, v.get<uint256_t>(idx)); break;
				case INT64: out.set({i, j}, v.get<int64_t>(idx)); break;
				default: out.set({i, j}, v.get<int>(idx)); break;
			}
		}
	}
	return out;
}

static void storePlane(NDArray& c, const NDArray& plane, int batch, int b) {
	int M = plane.shape.get(0);
	int N = plane.shape.get(1);
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			ArrayList<int> idx;
			idx.add(i);
			idx.add(j);
			if (batch > 1)
				idx.add(b);
			switch (c.type) {
				case F32: c.set(idx, plane.get<float>({i, j})); break;
				case F64: c.set(idx, plane.get<double>({i, j})); break;
				case UINT256: c.set(idx, plane.get<uint256_t>({i, j})); break;
				case INT64: c.set(idx, plane.get<int64_t>({i, j})); break;
				default: c.set(idx, plane.get<int>({i, j})); break;
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

	switch (a.type) {
		case F32:
			gemmFloat<float>((const float*)ap, (const float*)bp, (float*)cp, M, N, K, wrapA, wrapB);
			break;
		case F64:
			gemmFloat<double>((const double*)ap, (const double*)bp, (double*)cp, M, N, K, wrapA, wrapB);
			break;
		case INT8:
			gemmInt<int8_t, int32_t, int32_t>((const int8_t*)ap, (const int8_t*)bp, (int32_t*)cp,
			                                  M, N, K, wrapA, wrapB);
			break;
		case UINT8:
			gemmInt<uint8_t, int32_t, int32_t>((const uint8_t*)ap, (const uint8_t*)bp, (int32_t*)cp,
			                                   M, N, K, wrapA, wrapB);
			break;
		case INT32:
			gemmInt<int32_t, int64_t, int64_t>((const int32_t*)ap, (const int32_t*)bp, (int64_t*)cp,
			                                   M, N, K, wrapA, wrapB);
			break;
		case INT64:
			gemmInt64((const int64_t*)ap, (const int64_t*)bp, (int64_t*)cp, M, N, K, wrapA, wrapB);
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

NDArray ndmatmul(const NDArrayView& a, const NDArrayView& b) {
	if (a.getType() != b.getType())
		throw std::invalid_argument("NDArray::matmul - type mismatch");

	int ar = a.getShape().size();
	int br = b.getShape().size();
	int M = 0, N = 0, K = 0, K2 = 0, batch = 1;
	bool squeezeM = false;
	bool squeezeN = false;

	if (ar == 2 && br == 2) {
		M = a.getShape().get(0);
		K = a.getShape().get(1);
		K2 = b.getShape().get(0);
		N = b.getShape().get(1);
	} else if (ar == 2 && br == 1) {
		M = a.getShape().get(0);
		K = a.getShape().get(1);
		K2 = b.getShape().get(0);
		N = 1;
		squeezeN = true;
	} else if (ar == 1 && br == 2) {
		M = 1;
		K = a.getShape().get(0);
		K2 = b.getShape().get(0);
		N = b.getShape().get(1);
		squeezeM = true;
	} else if (ar == 3 && br == 3) {
		M = a.getShape().get(0);
		K = a.getShape().get(1);
		batch = a.getShape().get(2);
		K2 = b.getShape().get(0);
		N = b.getShape().get(1);
		if (b.getShape().get(2) != batch)
			throw std::invalid_argument("NDArray::matmul - batch mismatch");
	} else {
		throw std::invalid_argument("NDArray::matmul - rank must be 1, 2, or 3");
	}

	if (K != K2)
		throw std::invalid_argument("NDArray::matmul - inner dimension mismatch");

	ArrayList<int> outShape;
	if (!squeezeM)
		outShape.add(M);
	if (!squeezeN)
		outShape.add(N);
	if (batch > 1)
		outShape.add(batch);
	if (outShape.size() == 0)
		outShape = ArrayList<int>({});

	if (M <= 0 || N <= 0 || K < 0)
		return NDArray(outShape, matmulResultType(a.getType()));
	if (K == 0) {
		NDArray z(outShape, matmulResultType(a.getType()));
		return z;
	}

	bool wrapA = viewIsWrap(a);
	bool wrapB = viewIsWrap(b);
	NDArray acc({M, N}, matmulResultType(a.getType()));
	bool first = true;
	for (int bi = 0; bi < batch; ++bi) {
		NDArray ad = dense2d(a, M, K, batch, bi);
		NDArray bd = dense2d(b, K, N, batch, bi);
		NDArray plane = gemmOwned(ad, bd, wrapA, wrapB);
		if (batch == 1 && !squeezeM && !squeezeN && outShape.size() == 2)
			return plane;
		if (first) {
			acc = NDArray(outShape, plane.type);
			first = false;
		}
		if (batch > 1)
			storePlane(acc, plane, batch, bi);
		else {
			for (int i = 0; i < M; ++i) {
				for (int j = 0; j < N; ++j) {
					ArrayList<int> idx;
					if (!squeezeM)
						idx.add(i);
					if (!squeezeN)
						idx.add(j);
					if (idx.size() == 0) {
						acc.set({}, plane.get<float>({0, 0}));
						continue;
					}
					switch (plane.type) {
						case F32: acc.set(idx, plane.get<float>({i, j})); break;
						case F64: acc.set(idx, plane.get<double>({i, j})); break;
						case UINT256: acc.set(idx, plane.get<uint256_t>({i, j})); break;
						case INT64: acc.set(idx, plane.get<int64_t>({i, j})); break;
						default: acc.set(idx, plane.get<int>({i, j})); break;
					}
				}
			}
		}
	}
	return acc;
}
