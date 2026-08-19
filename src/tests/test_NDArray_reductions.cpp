
// Copyright 2021-2026 komozoi
// Original Creation Date: 2026-8-2
//
// All rights reserved.  Do not copy, distribute, or execute, in compiled or source form,
// any portion of this software, except software not created by me.
// If you find this software, please notify me immediately: komozoi@protonmail.com
//
//

#include <gtest/gtest.h>

#include "NDArray.h"


// ============================================================
// Full reductions (→ scalar, empty shape)
// ============================================================

TEST(NDArray_reductions, Sum_F32) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f, 4.0f}));
	NDArray s = a.sum();
	EXPECT_EQ(s.type, F32);
	EXPECT_EQ(s.shape.size(), 0);
	EXPECT_FLOAT_EQ(s.get<float>({}), 10.0f);
}

TEST(NDArray_reductions, Sum_UINT8_PromotesToINT32) {
	NDArray a(ArrayList<uint8_t>({200, 200, 200}));
	NDArray s = a.sum();
	EXPECT_EQ(s.type, INT32);
	EXPECT_EQ(s.shape.size(), 0);
	EXPECT_EQ(s.get<int32_t>({}), 600);
}

TEST(NDArray_reductions, Sum_BINARY) {
	NDArray b({5}, BINARY);
	b.set({0}, 1);
	b.set({2}, 1);
	b.set({4}, 1);
	NDArray s = b.sum();
	EXPECT_EQ(s.type, INT32);
	EXPECT_EQ(s.get<int32_t>({}), 3);
}

TEST(NDArray_reductions, Prod_F32_And_UINT8) {
	NDArray a(ArrayList({2.0f, 3.0f, 4.0f}));
	NDArray p = a.prod();
	EXPECT_EQ(p.type, F32);
	EXPECT_FLOAT_EQ(p.get<float>({}), 24.0f);

	NDArray u(ArrayList<uint8_t>({2, 3, 4}));
	NDArray up = u.prod();
	EXPECT_EQ(up.type, INT32);
	EXPECT_EQ(up.get<int32_t>({}), 24);
}

TEST(NDArray_reductions, Mean_F32_StaysF32_IntegersUseF64) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f, 4.0f}));
	NDArray m = a.mean();
	EXPECT_EQ(m.type, F32);
	EXPECT_FLOAT_EQ(m.get<float>({}), 2.5f);

	// Non-F32 inputs mean in F64 for better precision
	NDArray u(ArrayList<uint8_t>({2, 4, 6}));
	NDArray um = u.mean();
	EXPECT_EQ(um.type, F64);
	EXPECT_DOUBLE_EQ(um.get<double>({}), 4.0);
}

TEST(NDArray_reductions, MinMax_KeepType) {
	NDArray a(ArrayList({3.0f, -1.0f, 5.0f, 0.0f}));
	NDArray mn = a.min();
	NDArray mx = a.max();
	EXPECT_EQ(mn.type, F32);
	EXPECT_EQ(mx.type, F32);
	EXPECT_FLOAT_EQ(mn.get<float>({}), -1.0f);
	EXPECT_FLOAT_EQ(mx.get<float>({}), 5.0f);

	NDArray u(ArrayList<uint8_t>({10, 3, 200, 7}));
	EXPECT_EQ(u.min().type, UINT8);
	EXPECT_EQ(u.max().type, UINT8);
	EXPECT_EQ(u.min().get<uint8_t>({}), 3);
	EXPECT_EQ(u.max().get<uint8_t>({}), 200);

	NDArray s({4}, INT8);
	s.set({0}, -5);
	s.set({1}, -128);
	s.set({2}, 7);
	s.set({3}, 0);
	EXPECT_EQ(s.min().type, INT8);
	EXPECT_EQ(s.max().type, INT8);
	EXPECT_EQ(s.min().get<int>({ }), -128);
	EXPECT_EQ(s.max().get<int>({ }), 7);
	NDArray sum = s.sum();
	EXPECT_EQ(sum.type, INT32);
	EXPECT_EQ(sum.get<int32_t>({}), -5 - 128 + 7 + 0);
}

TEST(NDArray_reductions, ScalarInput) {
	NDArray s(F32, 7.0f);
	EXPECT_FLOAT_EQ(s.sum().get<float>({}), 7.0f);
	EXPECT_FLOAT_EQ(s.mean().get<float>({}), 7.0f);
	EXPECT_FLOAT_EQ(s.min().get<float>({}), 7.0f);
	EXPECT_FLOAT_EQ(s.prod().get<float>({}), 7.0f);
}


// ============================================================
// Axis reductions
// ============================================================

TEST(NDArray_reductions, Sum_Axis0_Matrix) {
	// shape [2, 3]:
	// 1 2 3
	// 4 5 6
	NDArray m({2, 3}, ArrayList({
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f
	}));

	NDArray s0 = m.sum(0);
	ASSERT_EQ(s0.shape.size(), 1);
	EXPECT_EQ(s0.shape.get(0), 3);
	EXPECT_EQ(s0.type, F32);
	EXPECT_FLOAT_EQ(s0.get<float>({0}), 5.0f);
	EXPECT_FLOAT_EQ(s0.get<float>({1}), 7.0f);
	EXPECT_FLOAT_EQ(s0.get<float>({2}), 9.0f);
}

TEST(NDArray_reductions, Sum_Axis1_Matrix) {
	NDArray m({2, 3}, ArrayList({
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f
	}));

	NDArray s1 = m.sum(1);
	ASSERT_EQ(s1.shape.size(), 1);
	EXPECT_EQ(s1.shape.get(0), 2);
	EXPECT_FLOAT_EQ(s1.get<float>({0}), 6.0f);
	EXPECT_FLOAT_EQ(s1.get<float>({1}), 15.0f);
}

TEST(NDArray_reductions, Mean_Axis) {
	NDArray m({2, 2}, ArrayList({
		2.0f, 4.0f,
		6.0f, 8.0f
	}));
	NDArray mean0 = m.mean(0);
	EXPECT_EQ(mean0.type, F32);
	EXPECT_FLOAT_EQ(mean0.get<float>({0}), 4.0f);
	EXPECT_FLOAT_EQ(mean0.get<float>({1}), 6.0f);

	NDArray mean1 = m.mean(1);
	EXPECT_FLOAT_EQ(mean1.get<float>({0}), 3.0f);
	EXPECT_FLOAT_EQ(mean1.get<float>({1}), 7.0f);
}

TEST(NDArray_reductions, MinMax_Axis) {
	NDArray m({2, 3}, ArrayList({
		1.0f, 9.0f, 3.0f,
		4.0f, 2.0f, 8.0f
	}));

	NDArray mn0 = m.min(0);
	EXPECT_FLOAT_EQ(mn0.get<float>({0}), 1.0f);
	EXPECT_FLOAT_EQ(mn0.get<float>({1}), 2.0f);
	EXPECT_FLOAT_EQ(mn0.get<float>({2}), 3.0f);

	NDArray mx1 = m.max(1);
	EXPECT_FLOAT_EQ(mx1.get<float>({0}), 9.0f);
	EXPECT_FLOAT_EQ(mx1.get<float>({1}), 8.0f);
}

TEST(NDArray_reductions, Prod_Axis) {
	NDArray m({2, 2}, ArrayList({
		2.0f, 3.0f,
		4.0f, 5.0f
	}));
	NDArray p0 = m.prod(0);
	EXPECT_FLOAT_EQ(p0.get<float>({0}), 8.0f);
	EXPECT_FLOAT_EQ(p0.get<float>({1}), 15.0f);

	NDArray p1 = m.prod(1);
	EXPECT_FLOAT_EQ(p1.get<float>({0}), 6.0f);
	EXPECT_FLOAT_EQ(p1.get<float>({1}), 20.0f);
}

TEST(NDArray_reductions, UINT8_Sum_Axis_Promotes) {
	NDArray m({2, 2}, ArrayList<uint8_t>({200, 200, 200, 200}));
	NDArray s = m.sum(0);
	EXPECT_EQ(s.type, INT32);
	ASSERT_EQ(s.shape.size(), 1);
	EXPECT_EQ(s.shape.get(0), 2);
	EXPECT_EQ(s.get<int32_t>({0}), 400);
	EXPECT_EQ(s.get<int32_t>({1}), 400);
}

TEST(NDArray_reductions, Tensor3D_Sum_Axis) {
	// shape [2, 2, 2], values 0..7 in order
	NDArray t({2, 2, 2}, F32);
	float v = 0.0f;
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
			for (int k = 0; k < 2; ++k)
				t.set({i, j, k}, v++);

	// sum over last axis → [2, 2]: (0+1)=1, (2+3)=5, (4+5)=9, (6+7)=13
	NDArray s2 = t.sum(2);
	ASSERT_EQ(s2.shape.size(), 2);
	EXPECT_EQ(s2.shape.get(0), 2);
	EXPECT_EQ(s2.shape.get(1), 2);
	EXPECT_FLOAT_EQ(s2.get<float>({0, 0}), 1.0f);
	EXPECT_FLOAT_EQ(s2.get<float>({0, 1}), 5.0f);
	EXPECT_FLOAT_EQ(s2.get<float>({1, 0}), 9.0f);
	EXPECT_FLOAT_EQ(s2.get<float>({1, 1}), 13.0f);

	// sum over axis 0 → [2, 2]: pairs (0,4)=4, (1,5)=6, (2,6)=8, (3,7)=10
	NDArray s0 = t.sum(0);
	EXPECT_FLOAT_EQ(s0.get<float>({0, 0}), 4.0f);
	EXPECT_FLOAT_EQ(s0.get<float>({0, 1}), 6.0f);
	EXPECT_FLOAT_EQ(s0.get<float>({1, 0}), 8.0f);
	EXPECT_FLOAT_EQ(s0.get<float>({1, 1}), 10.0f);
}


// ============================================================
// Error cases
// ============================================================

TEST(NDArray_reductions, AxisOutOfRange_Throws) {
	NDArray m({2, 3}, F32);
	EXPECT_THROW(m.sum(-1), std::out_of_range);
	EXPECT_THROW(m.sum(2), std::out_of_range);
	EXPECT_THROW(m.mean(5), std::out_of_range);
}

TEST(NDArray_reductions, Scalar_AxisReduce_Throws) {
	NDArray s(F32, 1.0f);
	EXPECT_THROW(s.sum(0), std::invalid_argument);
}

TEST(NDArray_reductions, OriginalUnchanged) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f}));
	(void)a.sum();
	(void)a.mean();
	(void)a.min();
	EXPECT_FLOAT_EQ(a.get<float>({0}), 1.0f);
	EXPECT_FLOAT_EQ(a.get<float>({2}), 3.0f);
}


// ============================================================
// sumAs / prodAs / meanAs — Acc is the template argument
// ============================================================

TEST(NDArray_reductions, SumAs_UINT8_IntoUint32AndUint64) {
	NDArray a(ArrayList<uint8_t>({200, 200, 200}));
	EXPECT_EQ(a.sumAs<uint32_t>(), 600u);
	EXPECT_EQ(a.sumAs<uint64_t>(), 600ull);
	EXPECT_EQ(a.sumAs<int32_t>(), 600);
	EXPECT_EQ(a.view().sumAs<uint32_t>(), 600u);
	// default lossless policy unchanged
	EXPECT_EQ(a.sum().type, INT32);
	EXPECT_EQ(a.sum().get<int32_t>({}), 600);
}

TEST(NDArray_reductions, SumAs_BINARY_IntoUint32) {
	NDArray b({8}, BINARY);
	b.set({0}, 1);
	b.set({3}, 1);
	b.set({7}, 1);
	EXPECT_EQ(b.sumAs<uint32_t>(), 3u);
	EXPECT_EQ(b.sumAs<int32_t>(), 3);
	EXPECT_EQ(b.sum().type, INT32);
}

TEST(NDArray_reductions, SumAs_INT3_SignedValuesNotWrap) {
	NDArray a({4}, INT3);
	a.set({0}, 3);
	a.set({1}, 3);
	a.set({2}, -4);
	a.set({3}, 1);
	// signed lanes: 3+3-4+1 = 3 (not nibble wrap)
	EXPECT_EQ(a.sumAs<int32_t>(), 3);
	EXPECT_EQ(a.sumAs<int64_t>(), 3);
	EXPECT_FLOAT_EQ(a.sumAs<float>(), 3.0f);
	EXPECT_EQ(a.sum().get<int32_t>({}), 3);
	EXPECT_EQ(a.prodAs<int32_t>(), 3 * 3 * -4 * 1);
}

TEST(NDArray_reductions, ProdAs_INT3_Widens) {
	NDArray a({2}, INT3);
	a.set({0}, 3);
	a.set({1}, 3);
	EXPECT_EQ(a.prodAs<int32_t>(), 9);
	NDArray wrapped = a * a; // wrap-mul stays 1
	EXPECT_EQ(wrapped.get<int>({0}), 1);
}

TEST(NDArray_reductions, SumAs_F32_MatchesNaive) {
	NDArray a({256}, F32);
	float naive = 0.0f;
	double naive64 = 0.0;
	for (int i = 0; i < 256; ++i) {
		float v = 0.25f * (float)((i % 7) - 3);
		a.set({i}, v);
		naive += v;
		naive64 += (double)v;
	}
	EXPECT_NEAR(a.sumAs<float>(), naive, 1e-5f);
	EXPECT_NEAR(a.sumAs<double>(), naive64, 1e-12);
	EXPECT_FLOAT_EQ(a.sum().get<float>({}), a.sumAs<float>());
}

TEST(NDArray_reductions, MeanAs_F32_AndInteger) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f, 4.0f}));
	EXPECT_FLOAT_EQ(a.meanAs<float>(), 2.5f);
	EXPECT_DOUBLE_EQ(a.meanAs<double>(), 2.5);

	NDArray u(ArrayList<uint8_t>({2, 4, 6}));
	EXPECT_DOUBLE_EQ(u.meanAs<double>(), 4.0);
	EXPECT_EQ(u.mean().type, F64);
}

TEST(NDArray_reductions, SumAs_WrapViewNoCopy) {
	alignas(4) float buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	NDArray a = NDArray::wrap(buf, sizeof(buf), {4}, F32);
	EXPECT_FLOAT_EQ(a.sumAs<float>(), 10.0f);
	EXPECT_EQ(a.data(), static_cast<const void*>(buf));
}

TEST(NDArray_reductions, ArgMaxMin_F32_INT32_INT3_UINT8) {
	NDArray a(ArrayList({3.0f, -1.0f, 5.0f, 5.0f, 0.0f}));
	NDArray imax = a.argmax();
	NDArray imin = a.argmin();
	EXPECT_EQ(imax.type, INT64);
	EXPECT_EQ(imax.shape.size(), 0);
	EXPECT_EQ(imax.get<int64_t>({}), 2); // first 5
	EXPECT_EQ(imin.get<int64_t>({}), 1);

	NDArray i32({4}, INT32);
	i32.set({0}, 10);
	i32.set({1}, -4);
	i32.set({2}, 7);
	i32.set({3}, -4);
	EXPECT_EQ(i32.argmax().get<int64_t>({}), 0);
	EXPECT_EQ(i32.argmin().get<int64_t>({}), 1);

	NDArray i3({5}, INT3);
	const int v3[] = {1, -4, 3, -2, 3};
	for (int i = 0; i < 5; ++i)
		i3.set({i}, v3[i]);
	EXPECT_EQ(i3.argmax().get<int64_t>({}), 2);
	EXPECT_EQ(i3.argmin().get<int64_t>({}), 1);

	NDArray u(ArrayList<uint8_t>({10, 3, 200, 7}));
	EXPECT_EQ(u.argmax().get<int64_t>({}), 2);
	EXPECT_EQ(u.argmin().get<int64_t>({}), 1);
}

TEST(NDArray_reductions, ArgMaxMin_Axis) {
	NDArray m({2, 3}, ArrayList({
		1.0f, 9.0f, 3.0f,
		4.0f, 2.0f, 8.0f
	}));
	NDArray a0 = m.argmax(0);
	EXPECT_EQ(a0.type, INT64);
	ASSERT_EQ(a0.shape.size(), 1);
	EXPECT_EQ(a0.shape.get(0), 3);
	EXPECT_EQ(a0.get<int64_t>({0}), 1);
	EXPECT_EQ(a0.get<int64_t>({1}), 0);
	EXPECT_EQ(a0.get<int64_t>({2}), 1);

	NDArray n1 = m.argmin(1);
	EXPECT_EQ(n1.get<int64_t>({0}), 0);
	EXPECT_EQ(n1.get<int64_t>({1}), 1);

	NDArray i3({2, 3}, INT3);
	const int vals[2][3] = {{-4, 2, 1}, {3, -1, 0}};
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 3; ++j)
			i3.set({i, j}, vals[i][j]);
	NDArray i3min0 = i3.argmin(0);
	EXPECT_EQ(i3min0.get<int64_t>({0}), 0);
	EXPECT_EQ(i3min0.get<int64_t>({1}), 1);
	EXPECT_EQ(i3min0.get<int64_t>({2}), 1);
}

TEST(NDArray_reductions, ArgMaxMin_ViewAndEmpty) {
	NDArray m({2, 3}, ArrayList({1.0f, 2.0f, 9.0f, 4.0f, 8.0f, 3.0f}));
	EXPECT_EQ(m.row(0).argmax().get<int64_t>({}), 2);
	EXPECT_EQ(m.col(1).argmax().get<int64_t>({}), 1);
	NDArray empty = NDArray::empty(F32);
	EXPECT_THROW(empty.argmax(), std::invalid_argument);
	EXPECT_THROW(m.argmax(-1), std::out_of_range);
	NDArray s(F32, 3.0f);
	EXPECT_EQ(s.argmax().get<int64_t>({}), 0);
	EXPECT_THROW(s.argmax(0), std::invalid_argument);
}

/** First flat index of min/max via getFlat (reference for packed scans). */
static int64_t refArgExt(const NDArray& a, bool wantMin) {
	const size_t n = a.numElements();
	int64_t best = 0;
	double bv = a.getFlat<double>(0);
	for (size_t i = 1; i < n; ++i) {
		double v = a.getFlat<double>(i);
		if (wantMin ? v < bv : v > bv) {
			bv = v;
			best = (int64_t)i;
		}
	}
	return best;
}

TEST(NDArray_reductions, ArgMaxMin_PackedLong_F32) {
	const int n = 1024;
	NDArray a({n}, F32);
	for (int i = 0; i < n; ++i)
		a.set({i}, (float)((i * 17) % 251) - 80.0f);
	a.set({16}, 200.0f);
	a.set({17}, 200.0f);
	a.set({32}, 200.0f);
	a.set({512}, -200.0f);
	a.set({513}, -200.0f);
	EXPECT_EQ(a.argmax().get<int64_t>({}), 16);
	EXPECT_EQ(a.argmin().get<int64_t>({}), 512);
	EXPECT_EQ(a.argmax().get<int64_t>({}), refArgExt(a, false));
	EXPECT_EQ(a.argmin().get<int64_t>({}), refArgExt(a, true));
}

TEST(NDArray_reductions, ArgMaxMin_PackedTypes) {
	NDArray i8({200}, INT8);
	for (int i = 0; i < 200; ++i)
		i8.set({i}, (i % 40) - 20);
	i8.set({64}, 127);
	i8.set({65}, 127);
	i8.set({3}, -128);
	EXPECT_EQ(i8.argmax().get<int64_t>({}), 64);
	EXPECT_EQ(i8.argmin().get<int64_t>({}), 3);

	NDArray u8(ArrayList<uint8_t>({1, 2, 9, 9, 0, 4}));
	EXPECT_EQ(u8.argmax().get<int64_t>({}), 2);
	EXPECT_EQ(u8.argmin().get<int64_t>({}), 4);

	NDArray i64({80}, INT64);
	for (int i = 0; i < 80; ++i)
		i64.set({i}, (int64_t)i - 10);
	i64.set({70}, (int64_t)1 << 40);
	EXPECT_EQ(i64.argmax().get<int64_t>({}), 70);
	EXPECT_EQ(i64.argmin().get<int64_t>({}), 0);

	NDArray d({50}, F64);
	for (int i = 0; i < 50; ++i)
		d.set({i}, (double)i);
	d.set({7}, 1e300);
	EXPECT_EQ(d.argmax().get<int64_t>({}), 7);
	EXPECT_EQ(d.argmin().get<int64_t>({}), 0);

	NDArray h({40}, F16);
	for (int i = 0; i < 40; ++i)
		h.set({i}, (float)(i - 5));
	h.set({12}, 100.0f);
	h.set({13}, 100.0f);
	EXPECT_EQ(h.argmax().get<int64_t>({}), 12);
	EXPECT_EQ(h.argmin().get<int64_t>({}), 0);

	NDArray bf({40}, BF16);
	for (int i = 0; i < 40; ++i)
		bf.set({i}, (float)(20 - i));
	EXPECT_EQ(bf.argmax().get<int64_t>({}), 0);
	EXPECT_EQ(bf.argmin().get<int64_t>({}), 39);
}

TEST(NDArray_reductions, ArgMaxMin_OffsetRowAndGappedCol) {
	NDArray m({8, 17}, F32);
	for (int r = 0; r < 8; ++r)
		for (int c = 0; c < 17; ++c)
			m.set({r, c}, (float)(r * 17 + c));
	m.set({3, 11}, 1000.0f);
	m.set({3, 12}, 1000.0f);
	m.set({5, 4}, -50.0f);
	EXPECT_TRUE(m.row(3).isContiguous());
	EXPECT_EQ(m.row(3).argmax().get<int64_t>({}), 11);
	EXPECT_EQ(m.row(5).argmin().get<int64_t>({}), 4);
	EXPECT_FALSE(m.col(4).isContiguous());
	EXPECT_EQ(m.col(4).argmin().get<int64_t>({}), 5);
	EXPECT_EQ(m.col(4).argmax().get<int64_t>({}), 7);

	NDArray i3({6, 20}, INT3);
	for (int r = 0; r < 6; ++r)
		for (int c = 0; c < 20; ++c)
			i3.set({r, c}, 0);
	i3.set({2, 9}, 3);
	i3.set({2, 10}, 3);
	i3.set({4, 1}, -4);
	EXPECT_EQ(i3.row(2).argmax().get<int64_t>({}), 9);
	EXPECT_EQ(i3.row(4).argmin().get<int64_t>({}), 1);
	EXPECT_EQ(i3.col(1).argmin().get<int64_t>({}), 4);
}

TEST(NDArray_reductions, ArgMaxMin_AxisLastPacked) {
	NDArray m({5, 65}, F32);
	for (int r = 0; r < 5; ++r)
		for (int c = 0; c < 65; ++c)
			m.set({r, c}, (float)c);
	m.set({2, 16}, 200.0f);
	m.set({2, 17}, 200.0f);
	NDArray ax = m.argmax(1);
	EXPECT_EQ(ax.type, INT64);
	ASSERT_EQ(ax.shape.size(), 1);
	EXPECT_EQ(ax.shape.get(0), 5);
	EXPECT_EQ(ax.get<int64_t>({0}), 64);
	EXPECT_EQ(ax.get<int64_t>({2}), 16);
	NDArray mn = m.argmin(1);
	EXPECT_EQ(mn.get<int64_t>({1}), 0);
	NDArray a0 = m.argmax(0);
	EXPECT_EQ(a0.shape.get(0), 65);
	EXPECT_EQ(a0.get<int64_t>({16}), 2);
}

TEST(NDArray_reductions, ArgMaxMin_BINARY_WordBoundary) {
	NDArray b({130}, BINARY);
	for (int i = 0; i < 130; ++i)
		b.set({i}, 0);
	b.set({63}, 1);
	b.set({64}, 1);
	EXPECT_EQ(b.argmax().get<int64_t>({}), 63);
	EXPECT_EQ(b.argmin().get<int64_t>({}), 0);
	NDArray ones({70}, BINARY);
	for (int i = 0; i < 70; ++i)
		ones.set({i}, 1);
	ones.set({67}, 0);
	EXPECT_EQ(ones.argmin().get<int64_t>({}), 67);
	EXPECT_EQ(ones.argmax().get<int64_t>({}), 0);
}

TEST(NDArray_reductions, SumAs_EmptyThrows) {
	NDArray a = NDArray::empty(F32);
	EXPECT_THROW(a.sumAs<float>(), std::invalid_argument);
	EXPECT_THROW(a.sum(), std::invalid_argument);
}
