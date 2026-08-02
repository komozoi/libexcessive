
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

TEST(NDArray_reductions, Sum_UINT8_PromotesToUINT256) {
	// Integer sums accumulate in UINT256 so they never silently wrap.
	NDArray a(ArrayList<uint8_t>({200, 200, 200}));
	NDArray s = a.sum();
	EXPECT_EQ(s.type, UINT256);
	EXPECT_EQ(s.shape.size(), 0);
	EXPECT_EQ(s.get<uint256_t>({}), uint256_t(600));
}

TEST(NDArray_reductions, Sum_BINARY) {
	NDArray b({5}, BINARY);
	b.set({0}, 1);
	b.set({2}, 1);
	b.set({4}, 1);
	NDArray s = b.sum();
	EXPECT_EQ(s.type, UINT256);
	EXPECT_EQ(s.get<uint256_t>({}), uint256_t(3));
}

TEST(NDArray_reductions, Prod_F32_And_UINT8) {
	NDArray a(ArrayList({2.0f, 3.0f, 4.0f}));
	NDArray p = a.prod();
	EXPECT_EQ(p.type, F32);
	EXPECT_FLOAT_EQ(p.get<float>({}), 24.0f);

	NDArray u(ArrayList<uint8_t>({2, 3, 4}));
	NDArray up = u.prod();
	EXPECT_EQ(up.type, UINT256);
	EXPECT_EQ(up.get<uint256_t>({}), uint256_t(24));
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
	EXPECT_EQ(s.type, UINT256);
	ASSERT_EQ(s.shape.size(), 1);
	EXPECT_EQ(s.shape.get(0), 2);
	EXPECT_EQ(s.get<uint256_t>({0}), uint256_t(400));
	EXPECT_EQ(s.get<uint256_t>({1}), uint256_t(400));
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
