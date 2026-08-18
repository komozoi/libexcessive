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
	// Tiling should not be slower than ~2x naive on this size (usually much faster).
	EXPECT_LT(tiledUs, naiveUs * 2 + 5000);
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
