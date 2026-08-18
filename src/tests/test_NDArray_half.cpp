// Copyright 2021-2026 komozoi
// F16 / BF16 kernel correctness (length, specials, score, matmul).

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

#include "NDArray.h"


static NDArray halfFromF32(const std::vector<float>& v, NDArrayType t) {
	NDArray a({(int)v.size()}, F32);
	for (size_t i = 0; i < v.size(); ++i)
		a.set({(int)i}, v[i]);
	return a.convert(t);
}

TEST(NDArray_half, ConvertSpecials_F16) {
	NDArray a({5}, F32);
	a.set({0}, 0.0f);
	a.set({1}, -0.0f);
	a.set({2}, std::numeric_limits<float>::infinity());
	a.set({3}, -std::numeric_limits<float>::infinity());
	a.set({4}, std::numeric_limits<float>::quiet_NaN());
	NDArray h = a.convert(F16);
	EXPECT_EQ(h.type, F16);
	EXPECT_FLOAT_EQ(h.get<float>({0}), 0.0f);
	EXPECT_FLOAT_EQ(h.get<float>({1}), 0.0f);
	EXPECT_TRUE(std::isinf(h.get<float>({2})));
	EXPECT_TRUE(std::isinf(h.get<float>({3})));
	EXPECT_TRUE(std::isnan(h.get<float>({4})));
	EXPECT_TRUE(h.get<float>({2}) > 0);
	EXPECT_TRUE(h.get<float>({3}) < 0);
}

TEST(NDArray_half, ConvertRoundTrip_Long) {
	NDArray a({257}, F32);
	for (int i = 0; i < 257; ++i)
		a.set({i}, (float)(i - 128) * 0.25f);
	NDArray h = a.convert(F16);
	NDArray back = h.convert(F32);
	for (int i = 0; i < 257; ++i)
		EXPECT_NEAR(back.get<float>({i}), a.get<float>({i}), 0.002f) << i;
	NDArray b = a.convert(BF16);
	NDArray bback = b.convert(F32);
	for (int i = 0; i < 257; ++i)
		EXPECT_NEAR(bback.get<float>({i}), a.get<float>({i}), 0.02f) << i;
}

TEST(NDArray_half, AddLong_StaysHalf) {
	NDArray a = halfFromF32({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0.5f}, F16);
	NDArray b = halfFromF32({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5f}, F16);
	NDArray c = a + b;
	EXPECT_EQ(c.type, F16);
	EXPECT_FLOAT_EQ(c.get<float>({0}), 2.0f);
	EXPECT_FLOAT_EQ(c.get<float>({15}), 17.0f);
	EXPECT_FLOAT_EQ(c.get<float>({16}), 1.0f);
	a += 1.0f;
	EXPECT_EQ(a.type, F16);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 2.0f);
}

TEST(NDArray_half, BF16_AddLong) {
	NDArray a = halfFromF32({1.5f, -2.0f, 8.0f, 0.0f, 4.0f, 4.0f, 4.0f, 4.0f}, BF16);
	NDArray b = halfFromF32({0.5f, 2.0f, -3.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, BF16);
	NDArray c = a + b;
	EXPECT_EQ(c.type, BF16);
	EXPECT_FLOAT_EQ(c.get<float>({0}), 2.0f);
	EXPECT_FLOAT_EQ(c.get<float>({1}), 0.0f);
	EXPECT_FLOAT_EQ(c.get<float>({2}), 5.0f);
}

TEST(NDArray_half, F16_plus_BF16_PromotesToF32) {
	NDArray a = halfFromF32({1.0f, 2.0f}, F16);
	NDArray b = halfFromF32({3.0f, 4.0f}, BF16);
	NDArray c = a + b;
	EXPECT_EQ(c.type, F32);
	EXPECT_FLOAT_EQ(c.get<float>({0}), 4.0f);
	EXPECT_FLOAT_EQ(c.get<float>({1}), 6.0f);
}

TEST(NDArray_half, SumDot_MatchesF32) {
	std::vector<float> va, vb;
	for (int i = 0; i < 128; ++i) {
		va.push_back((float)(i % 7) - 3.0f);
		vb.push_back((float)(i % 5) - 2.0f);
	}
	NDArray fa({128}, F32);
	NDArray fb({128}, F32);
	for (int i = 0; i < 128; ++i) {
		fa.set({i}, va[(size_t)i]);
		fb.set({i}, vb[(size_t)i]);
	}
	NDArray ha = fa.convert(F16);
	NDArray hb = fb.convert(F16);
	EXPECT_NEAR(ha.sum().get<float>({}), fa.sum().get<float>({}), 0.05f);
	EXPECT_NEAR(ha.dot<float>(hb), fa.dot<float>(fb), 0.15f);
	NDArray ba = fa.convert(BF16);
	NDArray bb = fb.convert(BF16);
	EXPECT_NEAR(ba.sum().get<float>({}), fa.sum().get<float>({}), 1.0f);
	EXPECT_NEAR(ba.dot<float>(bb), fa.dot<float>(fb), 2.0f);
}

TEST(NDArray_half, Matmul_F16_MatchesF32) {
	NDArray a({4, 3}, F32);
	NDArray b({3, 5}, F32);
	float v = 1.0f;
	for (int i = 0; i < 4; ++i)
		for (int k = 0; k < 3; ++k)
			a.set({i, k}, v++);
	v = 0.5f;
	for (int k = 0; k < 3; ++k)
		for (int j = 0; j < 5; ++j)
			b.set({k, j}, v++);
	NDArray cref = a.matmul(b);
	NDArray c = a.convert(F16).matmul(b.convert(F16));
	EXPECT_EQ(c.type, F32);
	ASSERT_EQ(c.shape.get(0), 4);
	ASSERT_EQ(c.shape.get(1), 5);
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 5; ++j)
			EXPECT_NEAR(c.get<float>({i, j}), cref.get<float>({i, j}), 0.05f);
}

TEST(NDArray_half, Gemv_BF16) {
	NDArray w({3, 4}, F32);
	NDArray x({4}, F32);
	for (int i = 0; i < 3; ++i)
		for (int k = 0; k < 4; ++k)
			w.set({i, k}, (float)(i + 1));
	for (int k = 0; k < 4; ++k)
		x.set({k}, 2.0f);
	NDArray y = w.convert(BF16).gemv(x.convert(BF16));
	EXPECT_EQ(y.type, F32);
	EXPECT_NEAR(y.get<float>({0}), 8.0f, 0.05f);
	EXPECT_NEAR(y.get<float>({1}), 16.0f, 0.05f);
	EXPECT_NEAR(y.get<float>({2}), 24.0f, 0.05f);
}

TEST(NDArray_half, Classify_Long) {
	NDArray a({64}, F16);
	for (int i = 0; i < 64; ++i)
		a.set({i}, (i == 10) ? std::numeric_limits<float>::quiet_NaN()
		     : (i == 20) ? std::numeric_limits<float>::infinity()
		     : (i == 30) ? 0.0f : 1.0f);
	EXPECT_EQ(a.countNonzero(), 63);
	NDArray nan = a.isNaN();
	EXPECT_EQ(nan.get<int>({10}), 1);
	EXPECT_EQ(nan.get<int>({0}), 0);
	NDArray inf = a.isInfinite();
	EXPECT_EQ(inf.get<int>({20}), 1);
	NDArray fin = a.isFinite();
	EXPECT_EQ(fin.get<int>({10}), 0);
	EXPECT_EQ(fin.get<int>({20}), 0);
	EXPECT_EQ(fin.get<int>({30}), 1);
}
