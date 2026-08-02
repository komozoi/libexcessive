
// Copyright 2021-2026 komozoi
// Original Creation Date: 2026-8-2
//
// All rights reserved.  Do not copy, distribute, or execute, in compiled or source form,
// any portion of this software, except software not created by me.
// If you find this software, please notify me immediately: komozoi@protonmail.com
//
//

#include <gtest/gtest.h>
#include <cmath>

#include "NDArray.h"


// ============================================================
// Unary element-wise
// ============================================================

TEST(NDArray_elementwise, Neg_F32) {
	NDArray a(ArrayList({1.0f, -2.0f, 0.0f}));
	NDArray b = a.neg();
	EXPECT_EQ(b.type, F32);
	EXPECT_FLOAT_EQ(b.get<float>({0}), -1.0f);
	EXPECT_FLOAT_EQ(b.get<float>({1}), 2.0f);
	EXPECT_FLOAT_EQ(b.get<float>({2}), 0.0f);
	// original unchanged
	EXPECT_FLOAT_EQ(a.get<float>({0}), 1.0f);
}

TEST(NDArray_elementwise, Neg_UnaryOperator) {
	NDArray a(ArrayList({3.0f, -4.0f}));
	NDArray b = -a;
	EXPECT_FLOAT_EQ(b.get<float>({0}), -3.0f);
	EXPECT_FLOAT_EQ(b.get<float>({1}), 4.0f);
}

TEST(NDArray_elementwise, Neg_UINT8_PromotesToF32) {
	NDArray a(ArrayList<uint8_t>({1, 2, 3}));
	NDArray b = a.neg();
	EXPECT_EQ(b.type, F32);
	EXPECT_FLOAT_EQ(b.get<float>({0}), -1.0f);
	EXPECT_FLOAT_EQ(b.get<float>({2}), -3.0f);
}

TEST(NDArray_elementwise, Abs_F32) {
	NDArray a(ArrayList({-1.5f, 2.0f, -0.0f}));
	NDArray b = a.abs();
	EXPECT_FLOAT_EQ(b.get<float>({0}), 1.5f);
	EXPECT_FLOAT_EQ(b.get<float>({1}), 2.0f);
}

TEST(NDArray_elementwise, Abs_UINT8_Identity) {
	NDArray a(ArrayList<uint8_t>({0, 5, 255}));
	NDArray b = a.abs();
	EXPECT_EQ(b.type, UINT8);
	EXPECT_EQ(b.get<uint8_t>({1}), 5);
	EXPECT_EQ(b.get<uint8_t>({2}), 255);
}

TEST(NDArray_elementwise, Sqrt_Exp_Log) {
	NDArray a(ArrayList({1.0f, 4.0f, 9.0f}));
	NDArray r = a.sqrt();
	EXPECT_FLOAT_EQ(r.get<float>({0}), 1.0f);
	EXPECT_FLOAT_EQ(r.get<float>({1}), 2.0f);
	EXPECT_FLOAT_EQ(r.get<float>({2}), 3.0f);

	NDArray e = NDArray(ArrayList({0.0f, 1.0f})).exp();
	EXPECT_FLOAT_EQ(e.get<float>({0}), 1.0f);
	EXPECT_NEAR(e.get<float>({1}), std::exp(1.0f), 1e-5f);

	NDArray l = NDArray(ArrayList({1.0f, std::exp(1.0f)})).log();
	EXPECT_NEAR(l.get<float>({0}), 0.0f, 1e-5f);
	EXPECT_NEAR(l.get<float>({1}), 1.0f, 1e-5f);
}

TEST(NDArray_elementwise, Sqrt_UINT8_PromotesToF32) {
	NDArray a(ArrayList<uint8_t>({4, 9, 16}));
	NDArray b = a.sqrt();
	EXPECT_EQ(b.type, F32);
	EXPECT_FLOAT_EQ(b.get<float>({0}), 2.0f);
	EXPECT_FLOAT_EQ(b.get<float>({1}), 3.0f);
	EXPECT_FLOAT_EQ(b.get<float>({2}), 4.0f);
}

TEST(NDArray_elementwise, Floor_Ceil_Round) {
	NDArray a(ArrayList({1.2f, 1.5f, 1.8f, -1.2f, -1.8f}));
	NDArray f = a.floor();
	EXPECT_FLOAT_EQ(f.get<float>({0}), 1.0f);
	EXPECT_FLOAT_EQ(f.get<float>({3}), -2.0f);

	NDArray c = a.ceil();
	EXPECT_FLOAT_EQ(c.get<float>({0}), 2.0f);
	EXPECT_FLOAT_EQ(c.get<float>({3}), -1.0f);

	NDArray r = a.round();
	EXPECT_FLOAT_EQ(r.get<float>({0}), 1.0f);
	EXPECT_FLOAT_EQ(r.get<float>({1}), 2.0f); // round half away from zero / banker's — libm round
	EXPECT_FLOAT_EQ(r.get<float>({2}), 2.0f);
}

TEST(NDArray_elementwise, Floor_OnInteger_IsCopy) {
	NDArray a(ArrayList<uint8_t>({1, 2, 3}));
	NDArray b = a.floor();
	EXPECT_EQ(b.type, UINT8);
	EXPECT_EQ(b.get<uint8_t>({2}), 3);
}


// ============================================================
// Binary element-wise: minimum / maximum / pow / mod
// ============================================================

TEST(NDArray_elementwise, Minimum_Maximum_F32) {
	NDArray a(ArrayList({1.0f, 5.0f, 3.0f}));
	NDArray b(ArrayList({4.0f, 2.0f, 3.0f}));

	NDArray mn = a.minimum(b);
	EXPECT_FLOAT_EQ(mn.get<float>({0}), 1.0f);
	EXPECT_FLOAT_EQ(mn.get<float>({1}), 2.0f);
	EXPECT_FLOAT_EQ(mn.get<float>({2}), 3.0f);

	NDArray mx = a.maximum(b);
	EXPECT_FLOAT_EQ(mx.get<float>({0}), 4.0f);
	EXPECT_FLOAT_EQ(mx.get<float>({1}), 5.0f);
	EXPECT_FLOAT_EQ(mx.get<float>({2}), 3.0f);
}

TEST(NDArray_elementwise, Minimum_Promotes_UINT8_F32) {
	NDArray a(ArrayList<uint8_t>({1, 10, 3}));
	NDArray b(ArrayList({2.5f, 2.5f, 2.5f}));
	NDArray mn = a.minimum(b);
	EXPECT_EQ(mn.type, F32);
	EXPECT_FLOAT_EQ(mn.get<float>({0}), 1.0f);
	EXPECT_FLOAT_EQ(mn.get<float>({1}), 2.5f);
	EXPECT_FLOAT_EQ(mn.get<float>({2}), 2.5f);
}

TEST(NDArray_elementwise, Pow_F32) {
	NDArray a(ArrayList({2.0f, 3.0f, 4.0f}));
	NDArray b(ArrayList({3.0f, 2.0f, 0.5f}));
	NDArray p = a.pow(b);
	EXPECT_FLOAT_EQ(p.get<float>({0}), 8.0f);
	EXPECT_FLOAT_EQ(p.get<float>({1}), 9.0f);
	EXPECT_FLOAT_EQ(p.get<float>({2}), 2.0f);
}

TEST(NDArray_elementwise, Pow_UINT8_Integer) {
	NDArray a(ArrayList<uint8_t>({2, 3, 4}));
	NDArray b(ArrayList<uint8_t>({3, 2, 1}));
	NDArray p = a.pow(b);
	EXPECT_EQ(p.type, UINT8);
	EXPECT_EQ(p.get<uint8_t>({0}), 8);
	EXPECT_EQ(p.get<uint8_t>({1}), 9);
	EXPECT_EQ(p.get<uint8_t>({2}), 4);
}

TEST(NDArray_elementwise, Mod_F32_And_UINT8) {
	NDArray a(ArrayList({5.5f, 7.0f}));
	NDArray b(ArrayList({2.0f, 3.0f}));
	NDArray m = a.mod(b);
	EXPECT_NEAR(m.get<float>({0}), std::fmod(5.5f, 2.0f), 1e-5f);
	EXPECT_FLOAT_EQ(m.get<float>({1}), 1.0f);

	NDArray u(ArrayList<uint8_t>({10, 17, 9}));
	NDArray v(ArrayList<uint8_t>({3, 5, 4}));
	NDArray um = u.mod(v);
	EXPECT_EQ(um.get<uint8_t>({0}), 1);
	EXPECT_EQ(um.get<uint8_t>({1}), 2);
	EXPECT_EQ(um.get<uint8_t>({2}), 1);
}

TEST(NDArray_elementwise, Scalar_Minimum_Pow) {
	NDArray a(ArrayList({1.0f, 5.0f, 3.0f}));
	NDArray mn = a.minimum(2.5f);
	EXPECT_FLOAT_EQ(mn.get<float>({0}), 1.0f);
	EXPECT_FLOAT_EQ(mn.get<float>({1}), 2.5f);

	NDArray p = a.pow(2.0f);
	EXPECT_FLOAT_EQ(p.get<float>({0}), 1.0f);
	EXPECT_FLOAT_EQ(p.get<float>({1}), 25.0f);
	EXPECT_FLOAT_EQ(p.get<float>({2}), 9.0f);
}


// ============================================================
// Comparisons → BINARY
// ============================================================

TEST(NDArray_elementwise, Compare_Equal_Less) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f}));
	NDArray b(ArrayList({1.0f, 0.0f, 5.0f}));

	NDArray eq = a.equal(b);
	EXPECT_EQ(eq.type, BINARY);
	EXPECT_EQ(eq.get<int>({0}), 1);
	EXPECT_EQ(eq.get<int>({1}), 0);
	EXPECT_EQ(eq.get<int>({2}), 0);

	NDArray lt = a.less(b);
	EXPECT_EQ(lt.get<int>({0}), 0);
	EXPECT_EQ(lt.get<int>({1}), 0);
	EXPECT_EQ(lt.get<int>({2}), 1);

	NDArray ge = a.greaterEqual(b);
	EXPECT_EQ(ge.get<int>({0}), 1);
	EXPECT_EQ(ge.get<int>({1}), 1);
	EXPECT_EQ(ge.get<int>({2}), 0);
}

TEST(NDArray_elementwise, Compare_Promotes_UINT8_F32) {
	NDArray a(ArrayList<uint8_t>({1, 2, 3}));
	NDArray b(ArrayList({1.0f, 2.5f, 3.0f}));
	NDArray eq = a.equal(b);
	EXPECT_EQ(eq.type, BINARY);
	EXPECT_EQ(eq.get<int>({0}), 1);
	EXPECT_EQ(eq.get<int>({1}), 0);
	EXPECT_EQ(eq.get<int>({2}), 1);
}

TEST(NDArray_elementwise, Compare_FloatScalar) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f}));
	NDArray m = a.greater(2.0f);
	EXPECT_EQ(m.get<int>({0}), 0);
	EXPECT_EQ(m.get<int>({1}), 0);
	EXPECT_EQ(m.get<int>({2}), 1);
}

TEST(NDArray_elementwise, Compare_DoesNotBreak_WholeArrayEquality) {
	NDArray a(ArrayList({1.0f, 2.0f}));
	NDArray b(ArrayList({1.0f, 2.0f}));
	NDArray c(ArrayList({1.0f, 3.0f}));
	EXPECT_TRUE(a == b);
	EXPECT_FALSE(a == c);
}


// ============================================================
// where / any / all
// ============================================================

TEST(NDArray_elementwise, Where_Basic) {
	NDArray cond(ArrayList({1.0f, 0.0f, 1.0f})); // truthy via non-BINARY
	NDArray x(ArrayList({10.0f, 20.0f, 30.0f}));
	NDArray y(ArrayList({1.0f, 2.0f, 3.0f}));

	NDArray r = NDArray::where(cond, x, y);
	EXPECT_EQ(r.type, F32);
	EXPECT_FLOAT_EQ(r.get<float>({0}), 10.0f);
	EXPECT_FLOAT_EQ(r.get<float>({1}), 2.0f);
	EXPECT_FLOAT_EQ(r.get<float>({2}), 30.0f);
}

TEST(NDArray_elementwise, Where_BINARY_Mask) {
	NDArray mask({3}, BINARY);
	mask.set({0}, 1);
	mask.set({1}, 0);
	mask.set({2}, 1);
	NDArray x(ArrayList<uint8_t>({9, 8, 7}));
	NDArray y(ArrayList<uint8_t>({1, 2, 3}));
	NDArray r = NDArray::where(mask, x, y);
	EXPECT_EQ(r.type, UINT8);
	EXPECT_EQ(r.get<uint8_t>({0}), 9);
	EXPECT_EQ(r.get<uint8_t>({1}), 2);
	EXPECT_EQ(r.get<uint8_t>({2}), 7);
}

TEST(NDArray_elementwise, Where_PromotesBranches) {
	NDArray mask({2}, BINARY);
	mask.set({0}, 1);
	mask.set({1}, 0);
	NDArray x(ArrayList<uint8_t>({1, 2}));
	NDArray y(ArrayList({3.5f, 4.5f}));
	NDArray r = NDArray::where(mask, x, y);
	EXPECT_EQ(r.type, F32);
	EXPECT_FLOAT_EQ(r.get<float>({0}), 1.0f);
	EXPECT_FLOAT_EQ(r.get<float>({1}), 4.5f);
}

TEST(NDArray_elementwise, Any_All) {
	NDArray zeros(ArrayList({0.0f, 0.0f, 0.0f}));
	NDArray mixed(ArrayList({0.0f, 1.0f, 0.0f}));
	NDArray ones(ArrayList({1.0f, 2.0f, 3.0f}));

	EXPECT_FALSE(zeros.any());
	EXPECT_FALSE(zeros.all());
	EXPECT_TRUE(mixed.any());
	EXPECT_FALSE(mixed.all());
	EXPECT_TRUE(ones.any());
	EXPECT_TRUE(ones.all());

	NDArray b({3}, BINARY);
	b.set({0}, 1);
	b.set({1}, 1);
	b.set({2}, 1);
	EXPECT_TRUE(b.all());
	b.set({1}, 0);
	EXPECT_FALSE(b.all());
	EXPECT_TRUE(b.any());
}
