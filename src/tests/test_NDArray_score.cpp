
// Copyright 2021-2026 komozoi
// Original Creation Date: 2026-8-17
//
// All rights reserved.

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>

#include "NDArray.h"


TEST(NDArray_score, Int3_DotWidens_NotWrapMul) {
	NDArray a({4}, INT3);
	NDArray b({4}, INT3);
	a.set({0}, 3);
	a.set({1}, 3);
	a.set({2}, -4);
	a.set({3}, 1);
	b.set({0}, 3);
	b.set({1}, -1);
	b.set({2}, -4);
	b.set({3}, 2);

	// Widening: 9 + (-3) + 16 + 2 = 24
	EXPECT_EQ(a.dot<int32_t>(b), 24);
	EXPECT_EQ(a.view().dot<int32_t>(b.view()), 24);
	EXPECT_FLOAT_EQ(a.dot<float>(b.view()), 24.0f);

	// operator* stays wrap-mul: 3*3 → 1, not 9
	NDArray prod = a * b;
	EXPECT_EQ(prod.type, INT3);
	EXPECT_EQ(prod.get<int>({0}), 1);
	EXPECT_EQ(prod.get<int>({1}), -3);
}

TEST(NDArray_score, F32_DotMatchesNaive) {
	NDArray a(ArrayList({1.0f, -2.5f, 3.0f, 0.25f}));
	NDArray b(ArrayList({2.0f, 4.0f, -1.0f, 8.0f}));
	float naive = 0;
	for (size_t i = 0; i < 4; ++i)
		naive += a.getFlat<float>(i) * b.getFlat<float>(i);
	EXPECT_NEAR(a.dot<float>(b), naive, 1e-6f);
	EXPECT_NEAR(a.view().dot<double>(b.view()), (double)naive, 1e-6);
}

TEST(NDArray_score, Uint8_DotWidens) {
	NDArray a({3}, UINT8);
	NDArray b({3}, UINT8);
	a.set({0}, 200);
	a.set({1}, 3);
	a.set({2}, 1);
	b.set({0}, 200);
	b.set({1}, 4);
	b.set({2}, 2);
	// 40000 + 12 + 2
	EXPECT_EQ(a.dot<int32_t>(b), 40014);
}

TEST(NDArray_score, Int8_DotWidens) {
	NDArray a({3}, INT8);
	NDArray b({3}, INT8);
	a.set({0}, 100);
	a.set({1}, -2);
	a.set({2}, 3);
	b.set({0}, -2);
	b.set({1}, 50);
	b.set({2}, 4);
	// -200 + -100 + 12
	EXPECT_EQ(a.dot<int32_t>(b), -288);
	EXPECT_EQ(a.l2Squared<int32_t>(), 100 * 100 + 4 + 9);
}

TEST(NDArray_score, Binary_DotIsPopcountAnd) {
	NDArray a({8}, BINARY);
	NDArray b({8}, BINARY);
	a.set({0}, 1);
	a.set({1}, 1);
	a.set({2}, 0);
	a.set({3}, 1);
	b.set({0}, 1);
	b.set({1}, 0);
	b.set({2}, 1);
	b.set({3}, 1);
	EXPECT_EQ(a.dot<int32_t>(b), 2);
}

TEST(NDArray_score, Hamming_MatchesReference) {
	NDArray a({16}, BINARY);
	NDArray b({16}, BINARY);
	for (int i = 0; i < 16; ++i) {
		a.set({i}, i % 3 == 0 ? 1 : 0);
		b.set({i}, i % 2 == 0 ? 1 : 0);
	}
	int ref = 0;
	for (int i = 0; i < 16; ++i)
		if (a.get<int>({i}) != b.get<int>({i}))
			++ref;
	EXPECT_EQ(a.hamming<int32_t>(b), ref);
	EXPECT_EQ(a.view().hamming<int32_t>(b.view()), ref);
}

TEST(NDArray_score, Hamming_WordAligned) {
	NDArray a({128}, BINARY);
	NDArray b({128}, BINARY);
	for (int i = 0; i < 128; ++i) {
		a.set({i}, (i & 1) ? 1 : 0);
		b.set({i}, (i & 2) ? 1 : 0);
	}
	int ref = 0;
	for (int i = 0; i < 128; ++i)
		if (a.get<int>({i}) != b.get<int>({i}))
			++ref;
	EXPECT_EQ(a.hamming<int64_t>(b), ref);
}

TEST(NDArray_score, Hamming_RejectsNonBinary) {
	NDArray a(ArrayList({1.0f, 2.0f}));
	NDArray b(ArrayList({1.0f, 0.0f}));
	EXPECT_THROW(a.hamming<int32_t>(b), std::invalid_argument);
}

TEST(NDArray_score, L2_SelfAndPair) {
	NDArray a(ArrayList({3.0f, 4.0f}));
	EXPECT_FLOAT_EQ(a.l2Squared<float>(), 25.0f);
	EXPECT_FLOAT_EQ(a.l2Norm<float>(), 5.0f);

	NDArray z(ArrayList({0.0f, 0.0f}));
	EXPECT_FLOAT_EQ(a.l2Squared<float>(z), 25.0f);
	EXPECT_FLOAT_EQ(a.l2Norm<float>(z), 5.0f);

	NDArray b(ArrayList({0.0f, 4.0f}));
	EXPECT_FLOAT_EQ(a.l2Squared<float>(b), 9.0f);
}

TEST(NDArray_score, Int3_L2Widens) {
	NDArray a({2}, INT3);
	a.set({0}, 3);
	a.set({1}, -4);
	// 9 + 16
	EXPECT_EQ(a.l2Squared<int32_t>(), 25);

	NDArray b({2}, INT3);
	b.set({0}, -3);
	b.set({1}, -4);
	// (3-(-3))^2 + 0 = 36
	EXPECT_EQ(a.l2Squared<int32_t>(b), 36);
}

TEST(NDArray_score, LengthMismatch_Throws) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f}));
	NDArray b(ArrayList({1.0f, 2.0f}));
	EXPECT_THROW(a.dot<float>(b), std::invalid_argument);
	EXPECT_THROW(a.l2Squared<float>(b), std::invalid_argument);
}

TEST(NDArray_score, MixedTypes_GenericPath) {
	NDArray i({3}, INT3);
	NDArray f(ArrayList({2.0f, 3.0f, -1.0f}));
	i.set({0}, 3);
	i.set({1}, 1);
	i.set({2}, -2);
	// 6 + 3 + 2
	EXPECT_FLOAT_EQ(i.dot<float>(f), 11.0f);
}

TEST(NDArray_score, CosineViaDotAndNorm) {
	NDArray a(ArrayList({1.0f, 0.0f}));
	NDArray b(ArrayList({1.0f, 0.0f}));
	float c = a.dot<float>(b) / (a.l2Norm<float>() * b.l2Norm<float>());
	EXPECT_NEAR(c, 1.0f, 1e-6f);
}

TEST(NDArray_score, WrapViews_ScoreWithoutCopy) {
	float xa[3] = {1.0f, 2.0f, 3.0f};
	float xb[3] = {4.0f, 5.0f, 6.0f};
	NDArrayView va = NDArrayView::wrap(xa, sizeof(xa), {3}, F32);
	NDArrayView vb = NDArrayView::wrap(xb, sizeof(xb), {3}, F32);
	EXPECT_FLOAT_EQ(va.dot<float>(vb), 32.0f);
}

TEST(NDArray_score, Long_F32_MatchesNaive) {
	const int n = 384;
	NDArray a({n}, F32);
	NDArray b({n}, F32);
	for (int i = 0; i < n; ++i) {
		a.set({i}, (float)(i % 17) * 0.25f - 2.0f);
		b.set({i}, (float)(i % 13) * 0.5f - 1.0f);
	}
	double dot = 0, l2s = 0, l2p = 0;
	for (int i = 0; i < n; ++i) {
		float x = a.getFlat<float>((size_t)i);
		float y = b.getFlat<float>((size_t)i);
		dot += (double)x * y;
		l2s += (double)x * x;
		double d = (double)x - y;
		l2p += d * d;
	}
	EXPECT_NEAR((double)a.dot<float>(b), dot, 1e-3);
	EXPECT_NEAR((double)a.l2Squared<float>(), l2s, 1e-3);
	EXPECT_NEAR((double)a.l2Squared<float>(b), l2p, 1e-3);
}

TEST(NDArray_score, Long_Uint8Int3Binary) {
	const int n = 256;
	NDArray ua({n}, UINT8), ub({n}, UINT8);
	NDArray ia({n}, INT3), ib({n}, INT3);
	NDArray ba({n}, BINARY), bb({n}, BINARY);
	int32_t uDot = 0, iDot = 0, ham = 0;
	for (int i = 0; i < n; ++i) {
		int av = i % 200, bv = (i * 3) % 180;
		ua.set({i}, av);
		ub.set({i}, bv);
		uDot += av * bv;
		int ai = (i % 8) - 4, bi = ((i * 5) % 8) - 4;
		ia.set({i}, ai);
		ib.set({i}, bi);
		iDot += ai * bi;
		int abit = i & 1, bbit = (i >> 1) & 1;
		ba.set({i}, abit);
		bb.set({i}, bbit);
		ham += abit ^ bbit;
	}
	EXPECT_EQ(ua.dot<int32_t>(ub), uDot);
	EXPECT_EQ(ia.dot<int32_t>(ib), iDot);
	EXPECT_EQ(ba.hamming<int32_t>(bb), ham);
	EXPECT_EQ(ba.dot<int32_t>(bb), n / 4); // both 1 when i%4==3? abit=i&1, bbit=(i>>1)&1 → AND when i%4==3
}
