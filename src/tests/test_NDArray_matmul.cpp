// Copyright 2021-2026 komozoi
// Original Creation Date: 2026-8-18
//
// All rights reserved.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

#include "NDArray.h"
#include "fs/FdHandle.h"


static void fillSeq(NDArray& a, int start) {
	const size_t n = a.numElements();
	for (size_t i = 0; i < n; ++i)
		a.setFlat(i, start + (int)i);
}

static NDArray naiveMatmulI64(const NDArray& a, const NDArray& b) {
	int M = a.shape.get(0);
	int K = a.shape.get(1);
	int N = b.shape.get(1);
	NDArrayType outT = INT64;
	if (a.type == F32)
		outT = F32;
	else if (a.type == F64)
		outT = F64;
	else if (a.type == UINT256)
		outT = UINT256;
	else if (a.type == INT64)
		outT = INT64;
	else if (a.type == INT32)
		outT = INT64;
	else
		outT = INT32;
	NDArray c({M, N}, outT);
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			if (a.type == F32) {
				float acc = 0;
				for (int k = 0; k < K; ++k)
					acc += a.get<float>({i, k}) * b.get<float>({k, j});
				c.set({i, j}, acc);
			} else if (a.type == F64) {
				double acc = 0;
				for (int k = 0; k < K; ++k)
					acc += a.get<double>({i, k}) * b.get<double>({k, j});
				c.set({i, j}, acc);
			} else if (a.type == UINT256) {
				uint256_t acc(0);
				for (int k = 0; k < K; ++k)
					acc = acc + a.get<uint256_t>({i, k}) * b.get<uint256_t>({k, j});
				c.set({i, j}, acc);
			} else if (a.type == INT64) {
				int64_t acc = 0;
				for (int k = 0; k < K; ++k)
					acc += a.get<int64_t>({i, k}) * b.get<int64_t>({k, j});
				c.set({i, j}, acc);
			} else if (a.type == INT32) {
				int64_t acc = 0;
				for (int k = 0; k < K; ++k)
					acc += (int64_t)a.get<int>({i, k}) * (int64_t)b.get<int>({k, j});
				c.set({i, j}, acc);
			} else {
				int64_t acc = 0;
				for (int k = 0; k < K; ++k)
					acc += (int64_t)a.get<int>({i, k}) * (int64_t)b.get<int>({k, j});
				c.set({i, j}, (int)acc);
			}
		}
	}
	return c;
}

static void expectClose(const NDArray& got, const NDArray& exp) {
	ASSERT_EQ(got.type, exp.type);
	ASSERT_EQ(got.shape.size(), exp.shape.size());
	for (int d = 0; d < got.shape.size(); ++d)
		ASSERT_EQ(got.shape.get(d), exp.shape.get(d));
	const size_t n = got.numElements();
	for (size_t i = 0; i < n; ++i) {
		if (got.type == F32)
			EXPECT_NEAR(got.getFlat<float>(i), exp.getFlat<float>(i), 1e-4f);
		else if (got.type == F64)
			EXPECT_NEAR(got.getFlat<double>(i), exp.getFlat<double>(i), 1e-10);
		else if (got.type == UINT256)
			EXPECT_EQ(got.getFlat<uint256_t>(i), exp.getFlat<uint256_t>(i));
		else if (got.type == INT64)
			EXPECT_EQ(got.getFlat<int64_t>(i), exp.getFlat<int64_t>(i));
		else
			EXPECT_EQ(got.getFlat<int>(i), exp.getFlat<int>(i));
	}
}

static ArrayList<size_t> packedStrides(const ArrayList<int>& shape) {
	size_t acc[NDArray::kMaxRank];
	int n = shape.size();
	size_t step = 1;
	for (int d = n - 1; d >= 0; --d) {
		acc[d] = step;
		int dim = shape.get(d);
		if (dim > 0)
			step *= (size_t)dim;
	}
	ArrayList<size_t> st;
	for (int d = 0; d < n; ++d)
		st.add(acc[d]);
	return st;
}

static NDArrayView packedOffsetView(const NDArray& owner, ArrayList<int> shape, size_t offset) {
	ArrayList<size_t> st = packedStrides(shape);
	return NDArrayView(owner.view().sharedBuffer(), shape, st, offset, owner.type);
}

static NDArray identitySquare(int n, NDArrayType t) {
	NDArray id({n, n}, t);
	for (int i = 0; i < n; ++i)
		id.set({i, i}, 1);
	return id;
}


TEST(NDArray_matmul, F32_SmallMatchesNaive) {
	NDArray a({2, 3}, F32);
	NDArray b({3, 2}, F32);
	a.set({0, 0}, 1.f); a.set({0, 1}, 2.f); a.set({0, 2}, 3.f);
	a.set({1, 0}, 4.f); a.set({1, 1}, 5.f); a.set({1, 2}, 6.f);
	b.set({0, 0}, 7.f); b.set({0, 1}, 8.f);
	b.set({1, 0}, 9.f); b.set({1, 1}, 10.f);
	b.set({2, 0}, 11.f); b.set({2, 1}, 12.f);
	NDArray c = a.matmul(b);
	EXPECT_EQ(c.type, F32);
	EXPECT_FLOAT_EQ(c.get<float>({0, 0}), 58.f);
	EXPECT_FLOAT_EQ(c.get<float>({0, 1}), 64.f);
	EXPECT_FLOAT_EQ(c.get<float>({1, 0}), 139.f);
	EXPECT_FLOAT_EQ(c.get<float>({1, 1}), 154.f);
}

TEST(NDArray_matmul, F64_Small) {
	NDArray a({2, 2}, F64);
	NDArray b({2, 2}, F64);
	a.set({0, 0}, 1.0); a.set({0, 1}, 0.0);
	a.set({1, 0}, 0.0); a.set({1, 1}, 1.0);
	b.set({0, 0}, 3.5); b.set({0, 1}, 4.5);
	b.set({1, 0}, 5.5); b.set({1, 1}, 6.5);
	NDArray c = a.matmul(b);
	EXPECT_EQ(c.type, F64);
	EXPECT_DOUBLE_EQ(c.get<double>({0, 0}), 3.5);
	EXPECT_DOUBLE_EQ(c.get<double>({1, 1}), 6.5);
}

TEST(NDArray_matmul, INT3_WidensNotWrap) {
	NDArray a({1, 1}, INT3);
	NDArray b({1, 1}, INT3);
	a.set({0, 0}, 3);
	b.set({0, 0}, 3);
	NDArray c = a.matmul(b);
	EXPECT_EQ(c.type, INT32);
	EXPECT_EQ(c.get<int>({0, 0}), 9);
}

TEST(NDArray_matmul, BINARY_AndPopcount) {
	NDArray a({1, 4}, BINARY);
	NDArray b({4, 1}, BINARY);
	a.set({0, 0}, 1); a.set({0, 1}, 1); a.set({0, 2}, 0); a.set({0, 3}, 1);
	b.set({0, 0}, 1); b.set({1, 0}, 0); b.set({2, 0}, 1); b.set({3, 0}, 1);
	NDArray c = a.matmul(b);
	EXPECT_EQ(c.type, INT32);
	EXPECT_EQ(c.get<int>({0, 0}), 2);
}

TEST(NDArray_matmul, INT8_UINT8_INT32) {
	NDArray a8({2, 2}, INT8);
	NDArray b8({2, 2}, INT8);
	a8.set({0, 0}, -2); a8.set({0, 1}, 3);
	a8.set({1, 0}, 4); a8.set({1, 1}, -1);
	b8.set({0, 0}, 5); b8.set({0, 1}, -3);
	b8.set({1, 0}, 2); b8.set({1, 1}, 7);
	expectClose(a8.matmul(b8), naiveMatmulI64(a8, b8));

	NDArray au({2, 3}, UINT8);
	NDArray bu({3, 2}, UINT8);
	fillSeq(au, 1);
	fillSeq(bu, 2);
	expectClose(au.matmul(bu), naiveMatmulI64(au, bu));

	NDArray ai({2, 2}, INT32);
	NDArray bi({2, 2}, INT32);
	fillSeq(ai, -3);
	fillSeq(bi, 4);
	NDArray ci = ai.matmul(bi);
	EXPECT_EQ(ci.type, INT64);
	expectClose(ci, naiveMatmulI64(ai, bi));
}

TEST(NDArray_matmul, INT64_AndUINT256) {
	NDArray a({2, 2}, INT64);
	NDArray b({2, 2}, INT64);
	a.set({0, 0}, (int64_t)2); a.set({0, 1}, (int64_t)0);
	a.set({1, 0}, (int64_t)0); a.set({1, 1}, (int64_t)3);
	b.set({0, 0}, (int64_t)4); b.set({0, 1}, (int64_t)5);
	b.set({1, 0}, (int64_t)6); b.set({1, 1}, (int64_t)7);
	expectClose(a.matmul(b), naiveMatmulI64(a, b));

	NDArray au({2, 2}, UINT256);
	NDArray bu({2, 2}, UINT256);
	au.set({0, 0}, uint256_t(2)); au.set({0, 1}, uint256_t(1));
	au.set({1, 0}, uint256_t(0)); au.set({1, 1}, uint256_t(3));
	bu.set({0, 0}, uint256_t(4)); bu.set({0, 1}, uint256_t(0));
	bu.set({1, 0}, uint256_t(1)); bu.set({1, 1}, uint256_t(5));
	expectClose(au.matmul(bu), naiveMatmulI64(au, bu));
}

TEST(NDArray_matmul, IdentityAllDenseTypes) {
	NDArrayType types[] = { F32, F64, INT8, UINT8, INT32, INT64 };
	for (NDArrayType t : types) {
		NDArray id({3, 3}, t);
		NDArray x({3, 2}, t);
		for (int i = 0; i < 3; ++i)
			id.set({i, i}, 1);
		fillSeq(x, 1);
		expectClose(id.matmul(x), naiveMatmulI64(id, x));
	}
}

TEST(NDArray_matmul, GemvMatchesN1) {
	NDArray a({4, 3}, F32);
	NDArray x({3}, F32);
	fillSeq(a, 1);
	x.set({0}, 1.f); x.set({1}, 2.f); x.set({2}, 3.f);
	NDArray y = a.gemv(x);
	NDArray b({3, 1}, F32);
	b.set({0, 0}, 1.f); b.set({1, 0}, 2.f); b.set({2, 0}, 3.f);
	NDArray c = a.matmul(b);
	ASSERT_EQ(y.numElements(), (size_t)4);
	for (int i = 0; i < 4; ++i)
		EXPECT_FLOAT_EQ(y.getFlat<float>((size_t)i), c.get<float>({i, 0}));
}

TEST(NDArray_matmul, EmptyAndThrows) {
	NDArray emptyA({0, 4}, F32);
	NDArray emptyB({4, 3}, F32);
	NDArray z = emptyA.matmul(emptyB);
	EXPECT_TRUE(z.isEmpty());

	NDArray a({2, 3}, F32);
	NDArray b({4, 2}, F32);
	EXPECT_THROW(a.matmul(b), std::invalid_argument);

	NDArray u({2, 2}, UINT8);
	NDArray f({2, 2}, F32);
	EXPECT_THROW(u.matmul(f), std::invalid_argument);
}

TEST(NDArray_matmul, WrapMatchesOwned) {
	float abuf[6] = {1, 2, 3, 4, 5, 6};
	float bbuf[6] = {7, 8, 9, 10, 11, 12};
	NDArray aw = NDArray::wrap(abuf, sizeof(abuf), {2, 3}, F32);
	NDArray bw = NDArray::wrap(bbuf, sizeof(bbuf), {3, 2}, F32);
	NDArray ao({2, 3}, F32);
	NDArray bo({3, 2}, F32);
	for (int i = 0; i < 6; ++i) {
		ao.setFlat((size_t)i, abuf[i]);
		bo.setFlat((size_t)i, bbuf[i]);
	}
	expectClose(aw.matmul(bw), ao.matmul(bo));
}

TEST(NDArray_matmul, ViewMatchesOwner) {
	NDArray a({2, 2}, F32);
	NDArray b({2, 2}, F32);
	fillSeq(a, 1);
	fillSeq(b, 3);
	expectClose(a.view().matmul(b.view()), a.matmul(b));
}

TEST(NDArray_matmul, F32_TiledBeatsNaive) {
	const int n = 192;
	NDArray a({n, n}, F32);
	NDArray b({n, n}, F32);
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j) {
			a.set({i, j}, (float)((i * 17 + j) % 13) * 0.1f);
			b.set({i, j}, (float)((j * 13 + i) % 11) * 0.1f);
		}

	NDArray tiled = a.matmul(b);

	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	NDArray tiled2 = a.matmul(b);
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	(void)tiled2;

	float* A = static_cast<float*>(a.data());
	float* B = static_cast<float*>(b.data());
	float* C = new float[(size_t)n * (size_t)n];
	std::chrono::steady_clock::time_point n0 = std::chrono::steady_clock::now();
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			float acc = 0;
			for (int k = 0; k < n; ++k)
				acc += A[i * n + k] * B[k * n + j];
			C[i * n + j] = acc;
		}
	}
	std::chrono::steady_clock::time_point n1 = std::chrono::steady_clock::now();

	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j)
			EXPECT_NEAR(tiled.get<float>({i, j}), C[i * n + j], 1e-2f);

	int64_t tiledUs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
	int64_t naiveUs = std::chrono::duration_cast<std::chrono::microseconds>(n1 - n0).count();
	delete[] C;
	// AVX-512 8×16 pack should beat a naive ijk by a wide margin on 192³.
	EXPECT_LT(tiledUs * 4, naiveUs + 2000)
		<< "tiled=" << tiledUs << "us naive=" << naiveUs << "us";
}

// 448³ > 80e6, so ndmRunNStrips uses the default ThreadPool.
TEST(NDArray_matmul, F32_ParallelStripsMatchIdentity) {
	const int n = 448;
	NDArray a({n, n}, F32);
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j)
			a.set({i, j}, (float)((i * 17 + j) % 13) * 0.1f);
	NDArray I = identitySquare(n, F32);
	expectClose(a.matmul(I), a);
}

TEST(NDArray_matmul, OddSizesMatchNaive) {
	NDArray a({17, 19}, F32);
	NDArray b({19, 23}, F32);
	fillSeq(a, 1);
	fillSeq(b, 2);
	expectClose(a.matmul(b), naiveMatmulI64(a, b));

	NDArray a8({15, 17}, INT8);
	NDArray b8({17, 13}, INT8);
	for (int i = 0; i < 15 * 17; ++i)
		a8.setFlat((size_t)i, (i % 7) - 3);
	for (int i = 0; i < 17 * 13; ++i)
		b8.setFlat((size_t)i, (i % 5) - 2);
	expectClose(a8.matmul(b8), naiveMatmulI64(a8, b8));

	NDArray au({16, 16}, UINT8);
	NDArray bu({16, 16}, UINT8);
	fillSeq(au, 3);
	fillSeq(bu, 9);
	expectClose(au.matmul(bu), naiveMatmulI64(au, bu));

	NDArray a3({16, 16}, INT3);
	NDArray b3({16, 16}, INT3);
	for (int i = 0; i < 256; ++i) {
		a3.setFlat((size_t)i, (i % 7) - 3);
		b3.setFlat((size_t)i, (i % 5) - 2);
	}
	expectClose(a3.matmul(b3), naiveMatmulI64(a3, b3));

	NDArray a3o({7, 11}, INT3);
	NDArray b3o({11, 9}, INT3);
	for (int i = 0; i < 77; ++i)
		a3o.setFlat((size_t)i, (i % 7) - 3);
	for (int i = 0; i < 99; ++i)
		b3o.setFlat((size_t)i, (i % 5) - 2);
	expectClose(a3o.matmul(b3o), naiveMatmulI64(a3o, b3o));

	NDArray ab({64, 64}, BINARY);
	NDArray bb({64, 64}, BINARY);
	for (int i = 0; i < 64 * 64; ++i) {
		ab.setFlat((size_t)i, (i * 3) & 1);
		bb.setFlat((size_t)i, (i * 5) & 1);
	}
	expectClose(ab.matmul(bb), naiveMatmulI64(ab, bb));

	NDArray ai({17, 18}, INT32);
	NDArray bi({18, 19}, INT32);
	fillSeq(ai, -4);
	fillSeq(bi, 3);
	expectClose(ai.matmul(bi), naiveMatmulI64(ai, bi));
}

TEST(NDArray_matmul, Dot1D) {
	NDArray a({3}, F32);
	NDArray b({3}, F32);
	a.set({0}, 1.f); a.set({1}, 2.f); a.set({2}, 3.f);
	b.set({0}, 4.f); b.set({1}, 5.f); b.set({2}, 6.f);
	NDArray c = a.matmul(b);
	EXPECT_EQ(c.shape.size(), 0);
	EXPECT_FLOAT_EQ(c.get<float>({}), 32.f);
}

TEST(NDArray_matmul, BatchLeadingMatchesNumPy) {
	NDArray a({2, 2, 3}, F32);
	NDArray b({2, 3, 2}, F32);
	for (int p = 0; p < 2; ++p)
		for (int i = 0; i < 2; ++i)
			for (int k = 0; k < 3; ++k)
				a.set({p, i, k}, (float)(p * 10 + i * 3 + k + 1));
	for (int p = 0; p < 2; ++p)
		for (int k = 0; k < 3; ++k)
			for (int j = 0; j < 2; ++j)
				b.set({p, k, j}, (float)(p * 7 + k * 2 + j + 1));
	NDArray c = a.matmul(b);
	ASSERT_EQ(c.shape.size(), 3);
	EXPECT_EQ(c.shape.get(0), 2);
	EXPECT_EQ(c.shape.get(1), 2);
	EXPECT_EQ(c.shape.get(2), 2);
	for (int p = 0; p < 2; ++p) {
		NDArray ap({2, 3}, F32);
		NDArray bp({3, 2}, F32);
		for (int i = 0; i < 2; ++i)
			for (int k = 0; k < 3; ++k)
				ap.set({i, k}, a.get<float>({p, i, k}));
		for (int k = 0; k < 3; ++k)
			for (int j = 0; j < 2; ++j)
				bp.set({k, j}, b.get<float>({p, k, j}));
		NDArray exp = ap.matmul(bp);
		for (int i = 0; i < 2; ++i)
			for (int j = 0; j < 2; ++j)
				EXPECT_FLOAT_EQ(c.get<float>({p, i, j}), exp.get<float>({i, j}));
	}
}

TEST(NDArray_matmul, BatchBroadcastsRank2) {
	NDArray a({2, 2, 2}, F32);
	NDArray b({2, 2}, F32);
	fillSeq(b, 1);
	for (int p = 0; p < 2; ++p)
		for (int i = 0; i < 2; ++i)
			for (int k = 0; k < 2; ++k)
				a.set({p, i, k}, (float)(p + i + k + 1));
	NDArray c = a.matmul(b);
	ASSERT_EQ(c.shape.get(0), 2);
	NDArray a0({2, 2}, F32);
	for (int i = 0; i < 2; ++i)
		for (int k = 0; k < 2; ++k)
			a0.set({i, k}, a.get<float>({0, i, k}));
	NDArray exp = a0.matmul(b);
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
			EXPECT_FLOAT_EQ(c.get<float>({0, i, j}), exp.get<float>({i, j}));
}

TEST(NDArray_matmul, ZeroKIsZeroResult) {
	NDArray a({2, 0}, F32);
	NDArray b({0, 3}, F32);
	NDArray c = a.matmul(b);
	EXPECT_EQ(c.shape.get(0), 2);
	EXPECT_EQ(c.shape.get(1), 3);
	EXPECT_EQ(c.numElements(), (size_t)6);
	EXPECT_FLOAT_EQ(c.get<float>({0, 0}), 0.f);
	EXPECT_FLOAT_EQ(c.get<float>({1, 2}), 0.f);
}

TEST(NDArray_matmul, INT3_LargerThan1x1) {
	NDArray a({2, 3}, INT3);
	NDArray b({3, 2}, INT3);
	int av[6] = {1, -2, 3, -4, 0, 2};
	int bv[6] = {3, -1, 2, 0, -3, 1};
	for (int i = 0; i < 6; ++i) {
		a.setFlat((size_t)i, av[i]);
		b.setFlat((size_t)i, bv[i]);
	}
	NDArray c = a.matmul(b);
	EXPECT_EQ(c.type, INT32);
	expectClose(c, naiveMatmulI64(a, b));
}

TEST(NDArray_matmul, INT3_GemvAndFatMatchNaive) {
	NDArray a1({1, 96}, INT3);
	NDArray b1({96, 80}, INT3);
	for (int i = 0; i < 96; ++i)
		a1.setFlat((size_t)i, (i % 7) - 3);
	for (int i = 0; i < 96 * 80; ++i)
		b1.setFlat((size_t)i, (i % 5) - 2);
	expectClose(a1.matmul(b1), naiveMatmulI64(a1, b1));

	NDArray at({80, 96}, INT3);
	NDArray x({96, 1}, INT3);
	for (int i = 0; i < 80 * 96; ++i)
		at.setFlat((size_t)i, (i % 7) - 3);
	for (int i = 0; i < 96; ++i)
		x.setFlat((size_t)i, (i % 5) - 2);
	expectClose(at.matmul(x), naiveMatmulI64(at, x));

	NDArray a8({8, 64}, INT3);
	NDArray b8({64, 80}, INT3);
	for (int i = 0; i < 8 * 64; ++i)
		a8.setFlat((size_t)i, (i % 7) - 3);
	for (int i = 0; i < 64 * 80; ++i)
		b8.setFlat((size_t)i, (i % 5) - 2);
	expectClose(a8.matmul(b8), naiveMatmulI64(a8, b8));

	NDArray ao({3, 17}, INT3);
	NDArray bo({17, 19}, INT3);
	for (int i = 0; i < 51; ++i)
		ao.setFlat((size_t)i, (i % 7) - 3);
	for (int i = 0; i < 17 * 19; ++i)
		bo.setFlat((size_t)i, (i % 5) - 2);
	expectClose(ao.matmul(bo), naiveMatmulI64(ao, bo));

	NDArray am({96, 80}, INT3);
	NDArray bm({80, 64}, INT3);
	for (int i = 0; i < 96 * 80; ++i)
		am.setFlat((size_t)i, (i % 7) - 3);
	for (int i = 0; i < 80 * 64; ++i)
		bm.setFlat((size_t)i, (i % 5) - 2);
	expectClose(am.matmul(bm), naiveMatmulI64(am, bm));
}

TEST(NDArray_matmul, INT3_PackedBTileSample) {
	const int M = 32;
	const int K = 1024;
	const int N = 1024;
	NDArray a({M, K}, INT3);
	NDArray b({K, N}, INT3);
	for (int i = 0; i < M * K; ++i)
		a.setFlat((size_t)i, (i % 7) - 3);
	for (int i = 0; i < K * N; ++i)
		b.setFlat((size_t)i, (i % 5) - 2);
	NDArray c = a.matmul(b);
	int spots[][2] = {{0, 0}, {0, 16}, {0, N - 1}, {M - 1, 0}, {15, 32}, {31, 1008}};
	for (int s = 0; s < 6; ++s) {
		int i = spots[s][0];
		int j = spots[s][1];
		int acc = 0;
		for (int k = 0; k < K; ++k)
			acc += a.get<int>({i, k}) * b.get<int>({k, j});
		EXPECT_EQ(c.get<int>({i, j}), acc) << "at " << i << "," << j;
	}
}

TEST(NDArray_matmul, INT3_StreamBeatsUnpack) {
	const int K = 2048;
	const int N = 2048;
	NDArray a({1, K}, INT3);
	NDArray b({K, N}, INT3);
	for (int i = 0; i < K; ++i)
		a.setFlat((size_t)i, (i % 7) - 3);
	for (int i = 0; i < K * N; ++i)
		b.setFlat((size_t)i, (i % 5) - 2);

	NDArray got = a.matmul(b);
	EXPECT_EQ(got.type, INT32);
	{
		int acc0 = 0;
		int acc16 = 0;
		int accN = 0;
		for (int k = 0; k < K; ++k) {
			int av = a.get<int>({0, k});
			acc0 += av * b.get<int>({k, 0});
			acc16 += av * b.get<int>({k, 16});
			accN += av * b.get<int>({k, N - 1});
		}
		EXPECT_EQ(got.get<int>({0, 0}), acc0);
		EXPECT_EQ(got.get<int>({0, 16}), acc16);
		EXPECT_EQ(got.get<int>({0, N - 1}), accN);
	}

	(void)a.matmul(b);
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	NDArray c2 = a.matmul(b);
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	(void)c2;

	int8_t* A = new int8_t[K];
	int8_t* B = new int8_t[(size_t)K * (size_t)N];
	int32_t* C = new int32_t[N];
	std::chrono::steady_clock::time_point u0 = std::chrono::steady_clock::now();
	for (int k = 0; k < K; ++k)
		A[k] = (int8_t)a.get<int>({0, k});
	for (int k = 0; k < K; ++k)
		for (int j = 0; j < N; ++j)
			B[(size_t)k * (size_t)N + (size_t)j] = (int8_t)b.get<int>({k, j});
	for (int j = 0; j < N; ++j) {
		int acc = 0;
		for (int k = 0; k < K; ++k)
			acc += (int)A[k] * (int)B[(size_t)k * (size_t)N + (size_t)j];
		C[j] = acc;
	}
	std::chrono::steady_clock::time_point u1 = std::chrono::steady_clock::now();
	EXPECT_EQ(got.get<int>({0, 0}), C[0]);
	delete[] A;
	delete[] B;
	delete[] C;

	int64_t streamUs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
	int64_t unpackUs = std::chrono::duration_cast<std::chrono::microseconds>(u1 - u0).count();
	// Stream GEMM should beat the unpack-and-dot baseline.
	EXPECT_LT(streamUs * 5, unpackUs + 2000)
		<< "stream=" << streamUs << "us unpack-dot=" << unpackUs << "us";
}

TEST(NDArray_matmul, BINARY_Larger) {
	NDArray a({2, 8}, BINARY);
	NDArray b({8, 2}, BINARY);
	for (int i = 0; i < 16; ++i)
		a.setFlat((size_t)i, i & 1);
	for (int i = 0; i < 16; ++i)
		b.setFlat((size_t)i, (i / 2) & 1);
	NDArray c = a.matmul(b);
	EXPECT_EQ(c.type, INT32);
	expectClose(c, naiveMatmulI64(a, b));
}

TEST(NDArray_matmul, NonContiguousView) {
	NDArray a({2, 3}, F32);
	NDArray b({3, 2}, F32);
	fillSeq(a, 1);
	fillSeq(b, 2);
	// Transpose of A: shape (3,2), strides (1, 3) over row-major (2,3).
	ArrayList<size_t> st;
	st.add(1);
	st.add(3);
	NDArrayView at(a.view().sharedBuffer(), ArrayList<int>({3, 2}), st, 0, F32);
	NDArray denseT({3, 2}, F32);
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 2; ++j)
			denseT.set({i, j}, a.get<float>({j, i}));
	NDArray br({2, 2}, F32);
	fillSeq(br, 3);
	expectClose(at.matmul(br), denseT.matmul(br));
}

TEST(NDArray_matmul, OffsetContig2d_TimesIdentity) {
	NDArray a({2, 2}, ArrayList({1.0f, 2.0f, 3.0f, 4.0f}));
	NDArrayView bot = packedOffsetView(a, ArrayList<int>({1, 2}), 2);
	EXPECT_TRUE(bot.isContiguous());
	EXPECT_EQ(bot.getOffset(), 2u);
	NDArray I = identitySquare(2, F32);
	NDArray got = bot.matmul(I);
	EXPECT_FLOAT_EQ(got.get<float>({0, 0}), 3.0f);
	EXPECT_FLOAT_EQ(got.get<float>({0, 1}), 4.0f);

	NDArray t({3, 2}, ArrayList({
		1.0f, 2.0f,
		3.0f, 4.0f,
		5.0f, 6.0f
	}));
	NDArrayView mid = packedOffsetView(t, ArrayList<int>({1, 2}), 2);
	NDArray midGot = mid.matmul(I);
	EXPECT_FLOAT_EQ(midGot.get<float>({0, 0}), 3.0f);
	EXPECT_FLOAT_EQ(midGot.get<float>({0, 1}), 4.0f);

	NDArray m({3, 4}, ArrayList({
		0.0f, 1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f, 7.0f,
		8.0f, 9.0f, 10.0f, 11.0f
	}));
	NDArrayView row = packedOffsetView(m, ArrayList<int>({4}), 4);
	NDArrayView block = row.reshape(ArrayList<int>({2, 2}));
	EXPECT_TRUE(block.isContiguous());
	EXPECT_EQ(block.getOffset(), 4u);
	expectClose(block.matmul(I), block.copy());
}

TEST(NDArray_matmul, OffsetContig2d_MatchesCopiedPanel) {
	NDArray a({4, 3}, F32);
	NDArray b({3, 2}, F32);
	fillSeq(a, 1);
	fillSeq(b, 10);
	NDArrayView av = packedOffsetView(a, ArrayList<int>({2, 3}), 6);
	EXPECT_TRUE(av.isContiguous());
	EXPECT_EQ(av.getOffset(), 6u);
	expectClose(av.matmul(b), av.copy().matmul(b));

	NDArray tallB({6, 2}, F32);
	fillSeq(tallB, 20);
	NDArrayView bv = packedOffsetView(tallB, ArrayList<int>({3, 2}), 6);
	NDArray left({2, 3}, F32);
	fillSeq(left, 1);
	expectClose(left.matmul(bv), left.matmul(bv.copy()));

	expectClose(av.matmul(bv), av.copy().matmul(bv.copy()));

	NDArray ai({4, 3}, INT32);
	NDArray bi({3, 2}, INT32);
	fillSeq(ai, -3);
	fillSeq(bi, 4);
	NDArrayView aiv = packedOffsetView(ai, ArrayList<int>({2, 3}), 6);
	expectClose(aiv.matmul(bi), aiv.copy().matmul(bi));
}

TEST(NDArray_matmul, OffsetContig2d_GemvShapes) {
	NDArray a({4, 3}, F32);
	fillSeq(a, 1);
	NDArrayView av = packedOffsetView(a, ArrayList<int>({2, 3}), 6);
	NDArray x({3}, F32);
	x.set({0}, 1.f); x.set({1}, 2.f); x.set({2}, 3.f);
	expectClose(av.gemv(x), av.copy().gemv(x));

	NDArray xcol({3, 1}, F32);
	xcol.set({0, 0}, 1.f); xcol.set({1, 0}, 2.f); xcol.set({2, 0}, 3.f);
	expectClose(av.matmul(xcol), av.copy().matmul(xcol));

	NDArray rowOwner({6}, F32);
	fillSeq(rowOwner, 2);
	NDArrayView rv = packedOffsetView(rowOwner, ArrayList<int>({3}), 3);
	NDArray B({3, 2}, F32);
	fillSeq(B, 5);
	expectClose(rv.matmul(B), rv.copy().matmul(B));
	expectClose(rv.matmul(rv), rv.copy().matmul(rv.copy()));
}

TEST(NDArray_matmul, OffsetNonContig2d_MatchesCopiedPanel) {
	NDArray a({2, 3}, ArrayList({
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f
	}));
	ArrayList<size_t> st;
	st.add(3);
	st.add(1);
	NDArrayView cols(a.view().sharedBuffer(), ArrayList<int>({2, 2}), st, 1, F32);
	EXPECT_FALSE(cols.isContiguous());
	EXPECT_EQ(cols.getOffset(), 1u);
	NDArray I = identitySquare(2, F32);
	expectClose(cols.matmul(I), cols.copy());
}

TEST(NDArray_matmul, OffsetContigBatch3d_MatchesCopiedPanel) {
	NDArray a({3, 2, 3}, F32);
	NDArray b({3, 3, 2}, F32);
	fillSeq(a, 1);
	fillSeq(b, 7);
	NDArrayView av = packedOffsetView(a, ArrayList<int>({2, 2, 3}), 6);
	NDArrayView bv = packedOffsetView(b, ArrayList<int>({2, 3, 2}), 6);
	EXPECT_TRUE(av.isContiguous());
	EXPECT_TRUE(bv.isContiguous());
	EXPECT_EQ(av.getOffset(), 6u);
	EXPECT_EQ(bv.getOffset(), 6u);

	NDArray I = identitySquare(3, F32);
	expectClose(av.matmul(I), av.copy().matmul(I));
	expectClose(av.copy().matmul(bv), av.copy().matmul(bv.copy()));
	expectClose(av.matmul(bv), av.copy().matmul(bv.copy()));

	NDArray b2({3, 2}, F32);
	fillSeq(b2, 3);
	expectClose(av.matmul(b2), av.copy().matmul(b2));

	NDArray a2({2, 3}, F32);
	fillSeq(a2, 4);
	expectClose(a2.matmul(bv), a2.matmul(bv.copy()));

	NDArray tallA({4, 3}, F32);
	fillSeq(tallA, 4);
	NDArrayView aOff = packedOffsetView(tallA, ArrayList<int>({2, 3}), 6);
	expectClose(aOff.matmul(b), aOff.copy().matmul(b));
}

TEST(NDArray_matmul, AdviseOnHeapAndMmap) {
	int x = 0;
	memoryAdviseWillNeed(&x, sizeof(x));
	memoryAdviseSequential(&x, sizeof(x));
	memoryAdviseHugePage(&x, sizeof(x));

	const size_t page = (size_t)sysconf(_SC_PAGESIZE);
	void* p = mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	ASSERT_NE(p, MAP_FAILED);
	memoryAdviseWillNeed(p, page);
	munmap(p, page);
}

static float refScaledGemv(const NDArray& w, const NDArray& x, const NDArray& scales, int groupSize,
                           int i) {
	const int M = w.shape.get(0);
	const int K = w.shape.get(1);
	(void)M;
	const int nGroups = (K + groupSize - 1) / groupSize;
	double acc = 0;
	for (int g = 0; g < nGroups; ++g) {
		const int k0 = g * groupSize;
		const int k1 = (k0 + groupSize < K) ? (k0 + groupSize) : K;
		int64_t dot = 0;
		for (int k = k0; k < k1; ++k)
			dot += (int64_t)w.get<int>({i, k}) * (int64_t)x.get<int>({k});
		float sc = (scales.shape.size() == 1)
			? scales.get<float>({i})
			: scales.get<float>({i, g});
		acc += (double)sc * (double)dot;
	}
	return (float)acc;
}

TEST(NDArray_matmul, GemvScaled_Group64_INT3) {
	const int M = 3;
	const int K = 128;
	const int gs = 64;
	NDArray w({M, K}, INT3);
	NDArray x({K}, INT3);
	NDArray scales({M, 2}, F32);
	for (int i = 0; i < M; ++i)
		for (int k = 0; k < K; ++k)
			w.set({i, k}, (k + i) % 7 - 3);
	for (int k = 0; k < K; ++k)
		x.set({k}, k % 5 - 2);
	scales.set({0, 0}, 0.5f);
	scales.set({0, 1}, 2.0f);
	scales.set({1, 0}, 1.0f);
	scales.set({1, 1}, -1.0f);
	scales.set({2, 0}, 0.25f);
	scales.set({2, 1}, 4.0f);

	NDArray y = w.gemv(x, scales, gs);
	EXPECT_EQ(y.type, F32);
	ASSERT_EQ(y.numElements(), (size_t)M);
	for (int i = 0; i < M; ++i)
		EXPECT_FLOAT_EQ(y.getFlat<float>((size_t)i), refScaledGemv(w, x, scales, gs, i));
}

TEST(NDArray_matmul, GemvScaled_PerRow_MatchesGemvThenMul) {
	NDArray w({4, 8}, UINT8);
	NDArray x({8}, UINT8);
	NDArray scales({4}, F32);
	for (int i = 0; i < 4; ++i)
		for (int k = 0; k < 8; ++k)
			w.set({i, k}, 10 + i + k);
	for (int k = 0; k < 8; ++k)
		x.set({k}, 1 + k);
	scales.set({0}, 0.5f);
	scales.set({1}, 1.0f);
	scales.set({2}, 2.0f);
	scales.set({3}, 0.25f);

	NDArray y = w.gemv(x, scales, 8);
	NDArray raw = w.gemv(x).convert(F32);
	for (int i = 0; i < 4; ++i)
		EXPECT_FLOAT_EQ(y.getFlat<float>((size_t)i),
		                raw.getFlat<float>((size_t)i) * scales.get<float>({i}));
}

TEST(NDArray_matmul, GemvScaled_UnscaledUnchanged) {
	NDArray a({4, 3}, F32);
	NDArray x({3}, F32);
	fillSeq(a, 1);
	x.set({0}, 1.f); x.set({1}, 2.f); x.set({2}, 3.f);
	NDArray y = a.gemv(x);
	EXPECT_EQ(y.type, F32);
	EXPECT_FLOAT_EQ(y.getFlat<float>(0), 1.f*1.f + 2.f*2.f + 3.f*3.f);
}

TEST(NDArray_matmul, GemvScaled_F32_Group) {
	NDArray w({2, 8}, F32);
	NDArray x({8}, F32);
	NDArray sc({2, 2}, F32);
	fillSeq(w, 1);
	fillSeq(x, 1);
	sc.set({0, 0}, 2.0f);
	sc.set({0, 1}, 0.5f);
	sc.set({1, 0}, 1.0f);
	sc.set({1, 1}, 3.0f);
	NDArray y = w.gemv(x, sc, 4);
	for (int i = 0; i < 2; ++i) {
		double acc = 0;
		for (int g = 0; g < 2; ++g) {
			double dot = 0;
			for (int k = g * 4; k < (g + 1) * 4; ++k)
				dot += (double)w.get<float>({i, k}) * (double)x.get<float>({k});
			acc += (double)sc.get<float>({i, g}) * dot;
		}
		EXPECT_FLOAT_EQ(y.getFlat<float>((size_t)i), (float)acc);
	}
}

TEST(NDArray_matmul, GemvScaled_LongK_INT3) {
	const int M = 4;
	const int K = 1024;
	const int gs = 64;
	NDArray w({M, K}, INT3);
	NDArray x({K}, INT3);
	NDArray sc({M, K / gs}, F32);
	for (int i = 0; i < M; ++i)
		for (int k = 0; k < K; ++k)
			w.set({i, k}, (k + i) % 7 - 3);
	for (int k = 0; k < K; ++k)
		x.set({k}, k % 5 - 2);
	for (int i = 0; i < M; ++i)
		for (int g = 0; g < K / gs; ++g)
			sc.set({i, g}, 0.25f * (float)(i + 1) * (g % 3 == 0 ? -1.f : 1.f));
	NDArray y = w.gemv(x, sc, gs);
	for (int i = 0; i < M; ++i)
		EXPECT_FLOAT_EQ(y.getFlat<float>((size_t)i), refScaledGemv(w, x, sc, gs, i));
}

TEST(NDArray_matmul, GemvScaled_Throws) {
	NDArray w({2, 8}, INT3);
	NDArray x({8}, INT3);
	NDArray sc({2, 1}, F32);
	EXPECT_THROW(w.gemv(x, sc, 0), std::invalid_argument);
	NDArray scBad({2, 3}, F32);
	EXPECT_THROW(w.gemv(x, scBad, 4), std::invalid_argument);
	NDArray scRow({2}, F32);
	EXPECT_THROW(w.gemv(x, scRow, 4), std::invalid_argument);
	NDArray xBad({7}, INT3);
	NDArray scOk({2, 2}, F32);
	EXPECT_THROW(w.gemv(xBad, scOk, 4), std::invalid_argument);
}
