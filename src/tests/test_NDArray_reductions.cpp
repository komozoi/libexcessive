
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

TEST(NDArray_reductions, SumAs_EmptyThrows) {
	NDArray a = NDArray::empty(F32);
	EXPECT_THROW(a.sumAs<float>(), std::invalid_argument);
	EXPECT_THROW(a.sum(), std::invalid_argument);
}
