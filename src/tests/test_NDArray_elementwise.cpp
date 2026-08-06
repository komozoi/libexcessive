
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
#include <limits>

#include "NDArray.h"


// ============================================================
// Unary element-wise — IN PLACE (operators copy)
// ============================================================

TEST(NDArray_elementwise, Neg_InPlace) {
	NDArray a(ArrayList({1.0f, -2.0f, 0.0f}));
	a.neg();
	EXPECT_EQ(a.type, F32);
	EXPECT_FLOAT_EQ(a.get<float>({0}), -1.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1}), 2.0f);
	EXPECT_FLOAT_EQ(a.get<float>({2}), 0.0f);
}

TEST(NDArray_elementwise, Neg_UnaryOperator_Copies) {
	NDArray a(ArrayList({3.0f, -4.0f}));
	NDArray b = -a;
	EXPECT_FLOAT_EQ(b.get<float>({0}), -3.0f);
	EXPECT_FLOAT_EQ(b.get<float>({1}), 4.0f);
	// original unchanged
	EXPECT_FLOAT_EQ(a.get<float>({0}), 3.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1}), -4.0f);
}

TEST(NDArray_elementwise, Neg_UINT8_PromotesToINT32) {
	NDArray a(ArrayList<uint8_t>({1, 2, 3}));
	a.neg();
	EXPECT_EQ(a.type, INT32);
	EXPECT_EQ(a.get<int32_t>({0}), -1);
	EXPECT_EQ(a.get<int32_t>({2}), -3);
}

TEST(NDArray_elementwise, Abs_F32_InPlace) {
	NDArray a(ArrayList({-1.5f, 2.0f, -0.0f}));
	a.abs();
	EXPECT_FLOAT_EQ(a.get<float>({0}), 1.5f);
	EXPECT_FLOAT_EQ(a.get<float>({1}), 2.0f);
}

TEST(NDArray_elementwise, Abs_UINT8_NoOp) {
	NDArray a(ArrayList<uint8_t>({0, 5, 255}));
	a.abs();
	EXPECT_EQ(a.type, UINT8);
	EXPECT_EQ(a.get<uint8_t>({1}), 5);
	EXPECT_EQ(a.get<uint8_t>({2}), 255);
}

TEST(NDArray_elementwise, Sqrt_Exp_Log) {
	NDArray a(ArrayList({1.0f, 4.0f, 9.0f}));
	a.sqrt();
	EXPECT_FLOAT_EQ(a.get<float>({0}), 1.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1}), 2.0f);
	EXPECT_FLOAT_EQ(a.get<float>({2}), 3.0f);

	NDArray e(ArrayList({0.0f, 1.0f}));
	e.exp();
	EXPECT_FLOAT_EQ(e.get<float>({0}), 1.0f);
	EXPECT_NEAR(e.get<float>({1}), std::exp(1.0f), 1e-5f);

	NDArray l(ArrayList({1.0f, std::exp(1.0f)}));
	l.log();
	EXPECT_NEAR(l.get<float>({0}), 0.0f, 1e-5f);
	EXPECT_NEAR(l.get<float>({1}), 1.0f, 1e-5f);
}

TEST(NDArray_elementwise, Sqrt_UINT8_PromotesToF64) {
	NDArray a(ArrayList<uint8_t>({4, 9, 16}));
	a.sqrt();
	EXPECT_EQ(a.type, F64);
	EXPECT_DOUBLE_EQ(a.get<double>({0}), 2.0);
	EXPECT_DOUBLE_EQ(a.get<double>({1}), 3.0);
	EXPECT_DOUBLE_EQ(a.get<double>({2}), 4.0);
}

TEST(NDArray_elementwise, Floor_Ceil_Round) {
	NDArray f(ArrayList({1.2f, 1.5f, 1.8f, -1.2f, -1.8f}));
	f.floor();
	EXPECT_FLOAT_EQ(f.get<float>({0}), 1.0f);
	EXPECT_FLOAT_EQ(f.get<float>({3}), -2.0f);

	NDArray c(ArrayList({1.2f, 1.5f, 1.8f, -1.2f, -1.8f}));
	c.ceil();
	EXPECT_FLOAT_EQ(c.get<float>({0}), 2.0f);
	EXPECT_FLOAT_EQ(c.get<float>({3}), -1.0f);

	NDArray r(ArrayList({1.2f, 1.5f, 1.8f, -1.2f, -1.8f}));
	r.round();
	EXPECT_FLOAT_EQ(r.get<float>({0}), 1.0f);
	EXPECT_FLOAT_EQ(r.get<float>({1}), 2.0f);
	EXPECT_FLOAT_EQ(r.get<float>({2}), 2.0f);
}

TEST(NDArray_elementwise, Floor_OnInteger_NoOp) {
	NDArray a(ArrayList<uint8_t>({1, 2, 3}));
	a.floor();
	EXPECT_EQ(a.type, UINT8);
	EXPECT_EQ(a.get<uint8_t>({2}), 3);
}

TEST(NDArray_elementwise, Square_InPlace_ChainsToPower4) {
	// b.square().square() → b**4 in place
	NDArray b(ArrayList({2.0f, 3.0f}));
	b.square().square();
	EXPECT_FLOAT_EQ(b.get<float>({0}), 16.0f); // 2^4
	EXPECT_FLOAT_EQ(b.get<float>({1}), 81.0f); // 3^4
}

TEST(NDArray_elementwise, MulOperator_StillCopies) {
	NDArray a(ArrayList({2.0f, 3.0f}));
	NDArray c = a * a;
	EXPECT_FLOAT_EQ(c.get<float>({0}), 4.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 2.0f); // original intact
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

TEST(NDArray_elementwise, Modulo_Operator) {
	NDArray a(ArrayList({10.0f, 17.0f, 9.0f}));
	NDArray b(ArrayList({3.0f, 5.0f, 4.0f}));

	NDArray m = a % b;
	EXPECT_FLOAT_EQ(m.get<float>({0}), 1.0f);
	EXPECT_FLOAT_EQ(m.get<float>({1}), 2.0f);
	EXPECT_FLOAT_EQ(m.get<float>({2}), 1.0f);

	NDArray s = a % 4.0f;
	EXPECT_FLOAT_EQ(s.get<float>({0}), 2.0f);
	EXPECT_FLOAT_EQ(s.get<float>({1}), 1.0f);
	EXPECT_FLOAT_EQ(s.get<float>({2}), 1.0f);

	a %= b;
	EXPECT_FLOAT_EQ(a.get<float>({0}), 1.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1}), 2.0f);
	EXPECT_FLOAT_EQ(a.get<float>({2}), 1.0f);
}

TEST(NDArray_elementwise, Log2_Log10_Log1p_Expm1) {
	NDArray a(ArrayList({1.0f, 8.0f, 100.0f}));
	NDArray a2 = a;
	a2.log2();
	EXPECT_NEAR(a2.get<float>({0}), 0.0f, 1e-5f);
	EXPECT_NEAR(a2.get<float>({1}), 3.0f, 1e-5f);

	NDArray a3 = a;
	a3.log10();
	EXPECT_NEAR(a3.get<float>({0}), 0.0f, 1e-5f);
	EXPECT_NEAR(a3.get<float>({2}), 2.0f, 1e-5f);

	NDArray x(ArrayList({0.0f, std::exp(1.0f) - 1.0f}));
	x.log1p();
	EXPECT_NEAR(x.get<float>({0}), 0.0f, 1e-5f);
	EXPECT_NEAR(x.get<float>({1}), 1.0f, 1e-4f);

	NDArray z(ArrayList({0.0f, 1.0f}));
	z.expm1();
	EXPECT_NEAR(z.get<float>({0}), 0.0f, 1e-5f);
	EXPECT_NEAR(z.get<float>({1}), std::exp(1.0f) - 1.0f, 1e-5f);
}

TEST(NDArray_elementwise, Sign_Square_Cbrt_Trig) {
	NDArray a(ArrayList({-2.0f, 0.0f, 3.0f}));
	NDArray s = a; // copy for sign
	s.sign();
	EXPECT_FLOAT_EQ(s.get<float>({0}), -1.0f);
	EXPECT_FLOAT_EQ(s.get<float>({1}), 0.0f);
	EXPECT_FLOAT_EQ(s.get<float>({2}), 1.0f);

	NDArray sq = a;
	sq.square();
	EXPECT_FLOAT_EQ(sq.get<float>({0}), 4.0f);
	EXPECT_FLOAT_EQ(sq.get<float>({2}), 9.0f);

	NDArray c(ArrayList({8.0f, 27.0f}));
	c.cbrt();
	EXPECT_NEAR(c.get<float>({0}), 2.0f, 1e-5f);
	EXPECT_NEAR(c.get<float>({1}), 3.0f, 1e-5f);

	const float halfPi = std::acos(-1.0f) * 0.5f;
	NDArray ang(ArrayList({0.0f, halfPi}));
	ang.sin();
	EXPECT_NEAR(ang.get<float>({0}), 0.0f, 1e-5f);
	EXPECT_NEAR(ang.get<float>({1}), 1.0f, 1e-5f);
}

TEST(NDArray_elementwise, InverseTrig) {
	NDArray a(ArrayList({0.0f, 1.0f, -1.0f}));
	NDArray as = a;
	as.asin();
	EXPECT_NEAR(as.get<float>({0}), 0.0f, 1e-5f);
	EXPECT_NEAR(as.get<float>({1}), std::asin(1.0f), 1e-5f);
	EXPECT_NEAR(as.get<float>({2}), std::asin(-1.0f), 1e-5f);

	NDArray ac(ArrayList({1.0f, 0.0f}));
	ac.acos();
	EXPECT_NEAR(ac.get<float>({0}), 0.0f, 1e-5f);
	EXPECT_NEAR(ac.get<float>({1}), std::acos(0.0f), 1e-5f);

	NDArray at(ArrayList({0.0f, 1.0f}));
	at.atan();
	EXPECT_NEAR(at.get<float>({0}), 0.0f, 1e-5f);
	EXPECT_NEAR(at.get<float>({1}), std::atan(1.0f), 1e-5f);
}

TEST(NDArray_elementwise, Hyperbolic) {
	NDArray a(ArrayList({0.0f, 1.0f}));
	NDArray sh = a;
	sh.sinh();
	EXPECT_NEAR(sh.get<float>({0}), 0.0f, 1e-5f);
	EXPECT_NEAR(sh.get<float>({1}), std::sinh(1.0f), 1e-5f);

	NDArray ch = a;
	ch.cosh();
	EXPECT_NEAR(ch.get<float>({0}), 1.0f, 1e-5f);
	EXPECT_NEAR(ch.get<float>({1}), std::cosh(1.0f), 1e-5f);

	NDArray th = a;
	th.tanh();
	EXPECT_NEAR(th.get<float>({0}), 0.0f, 1e-5f);
	EXPECT_NEAR(th.get<float>({1}), std::tanh(1.0f), 1e-5f);

	NDArray ash(ArrayList({0.0f, 1.0f}));
	ash.asinh();
	EXPECT_NEAR(ash.get<float>({0}), 0.0f, 1e-5f);
	EXPECT_NEAR(ash.get<float>({1}), std::asinh(1.0f), 1e-5f);
}

TEST(NDArray_elementwise, DegRad) {
	NDArray d(ArrayList({0.0f, 180.0f, 90.0f}));
	d.deg2rad();
	EXPECT_NEAR(d.get<float>({0}), 0.0f, 1e-5f);
	EXPECT_NEAR(d.get<float>({1}), (float)std::acos(-1.0), 1e-5f);
	EXPECT_NEAR(d.get<float>({2}), (float)std::acos(-1.0) * 0.5f, 1e-5f);

	NDArray r(ArrayList({0.0f, (float)std::acos(-1.0)}));
	r.rad2deg();
	EXPECT_NEAR(r.get<float>({0}), 0.0f, 1e-5f);
	EXPECT_NEAR(r.get<float>({1}), 180.0f, 1e-4f);
}

TEST(NDArray_elementwise, Atan2_Hypot) {
	NDArray y(ArrayList({0.0f, 1.0f, 1.0f}));
	NDArray x(ArrayList({1.0f, 0.0f, 1.0f}));
	NDArray a = y.atan2(x);
	EXPECT_NEAR(a.get<float>({0}), 0.0f, 1e-5f);
	EXPECT_NEAR(a.get<float>({1}), (float)std::acos(-1.0) * 0.5f, 1e-5f);
	EXPECT_NEAR(a.get<float>({2}), (float)std::atan2(1.0, 1.0), 1e-5f);

	// inputs not mutated
	EXPECT_FLOAT_EQ(y.get<float>({1}), 1.0f);

	NDArray h = y.hypot(x);
	EXPECT_NEAR(h.get<float>({0}), 1.0f, 1e-5f);
	EXPECT_NEAR(h.get<float>({1}), 1.0f, 1e-5f);
	EXPECT_NEAR(h.get<float>({2}), std::sqrt(2.0f), 1e-5f);

	NDArray h2 = y.hypot(0.0);
	EXPECT_NEAR(h2.get<float>({0}), 0.0f, 1e-5f);
	EXPECT_NEAR(h2.get<float>({1}), 1.0f, 1e-5f);
}

TEST(NDArray_elementwise, Clip) {
	NDArray a(ArrayList({-1.0f, 0.5f, 2.0f, 5.0f}));
	NDArray c = a.clip(0.0f, 2.0f);
	EXPECT_FLOAT_EQ(c.get<float>({0}), 0.0f);
	EXPECT_FLOAT_EQ(c.get<float>({1}), 0.5f);
	EXPECT_FLOAT_EQ(c.get<float>({2}), 2.0f);
	EXPECT_FLOAT_EQ(c.get<float>({3}), 2.0f);

	NDArray lo(ArrayList({0.0f, 0.0f, 1.0f, 1.0f}));
	NDArray hi(ArrayList({1.0f, 1.0f, 3.0f, 3.0f}));
	NDArray c2 = a.clip(lo, hi);
	EXPECT_FLOAT_EQ(c2.get<float>({0}), 0.0f);
	EXPECT_FLOAT_EQ(c2.get<float>({1}), 0.5f);
	EXPECT_FLOAT_EQ(c2.get<float>({2}), 2.0f);
	EXPECT_FLOAT_EQ(c2.get<float>({3}), 3.0f);
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
	NDArray m = a > 2.0f;
	EXPECT_EQ(m.get<int>({0}), 0);
	EXPECT_EQ(m.get<int>({1}), 0);
	EXPECT_EQ(m.get<int>({2}), 1);

	NDArray eq0 = a == 2.0f;
	EXPECT_EQ(eq0.get<int>({0}), 0);
	EXPECT_EQ(eq0.get<int>({1}), 1);
	EXPECT_EQ(eq0.get<int>({2}), 0);
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
	NDArray cond(ArrayList({1.0f, 0.0f, 1.0f}));
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

TEST(NDArray_elementwise, IsFinite_IsInfinite) {
	NDArray a(ArrayList({1.0f, std::numeric_limits<float>::infinity(),
	                     std::numeric_limits<float>::quiet_NaN(), -2.0f}));
	NDArray fin = a.isFinite();
	NDArray inf = a.isInfinite();
	EXPECT_EQ(fin.type, BINARY);
	EXPECT_EQ(inf.type, BINARY);
	EXPECT_EQ(fin.get<int>({0}), 1);
	EXPECT_EQ(fin.get<int>({1}), 0);
	EXPECT_EQ(fin.get<int>({2}), 0); // NaN is not finite
	EXPECT_EQ(fin.get<int>({3}), 1);
	EXPECT_EQ(inf.get<int>({0}), 0);
	EXPECT_EQ(inf.get<int>({1}), 1);
	EXPECT_EQ(inf.get<int>({2}), 0); // NaN is not infinite
	EXPECT_EQ(inf.get<int>({3}), 0);

	NDArray nan = a.isNaN();
	EXPECT_EQ(nan.type, BINARY);
	EXPECT_EQ(nan.get<int>({0}), 0);
	EXPECT_EQ(nan.get<int>({1}), 0);
	EXPECT_EQ(nan.get<int>({2}), 1);
	EXPECT_EQ(nan.get<int>({3}), 0);
	EXPECT_TRUE(a.isNaN().any());
	EXPECT_FALSE(a.isNaN().all());

	EXPECT_FALSE(a.isFinite().all());
	EXPECT_TRUE(a.isInfinite().any());
	EXPECT_FALSE(a.isInfinite().all());

	// Integers are always finite / never NaN
	NDArray i(ArrayList<int32_t>({1, 2, 3}));
	EXPECT_TRUE(i.isFinite().all());
	EXPECT_FALSE(i.isInfinite().any());
	EXPECT_FALSE(i.isNaN().any());

	NDArray d(ArrayList<double>({0.0, -std::numeric_limits<double>::infinity()}));
	EXPECT_EQ(d.isFinite().get<int>({0}), 1);
	EXPECT_EQ(d.isFinite().get<int>({1}), 0);
	EXPECT_EQ(d.isInfinite().get<int>({1}), 1);

	// Long vector: exercise word-packed / AVX-512 path (≥ 64 lanes)
	const int N = 130;
	NDArray longv({N}, F32);
	for (int i = 0; i < N; ++i) {
		if (i == 10)
			longv.set({i}, std::numeric_limits<float>::infinity());
		else if (i == 70)
			longv.set({i}, std::numeric_limits<float>::quiet_NaN());
		else
			longv.set({i}, (float)i);
	}
	NDArray lf = longv.isFinite();
	NDArray li = longv.isInfinite();
	NDArray ln = longv.isNaN();
	for (int i = 0; i < N; ++i) {
		EXPECT_EQ(lf.get<int>({i}), (i == 10 || i == 70) ? 0 : 1) << i;
		EXPECT_EQ(li.get<int>({i}), i == 10 ? 1 : 0) << i;
		EXPECT_EQ(ln.get<int>({i}), i == 70 ? 1 : 0) << i;
	}
	EXPECT_TRUE(NDArray::ones({5}).isFinite().all());
	EXPECT_FALSE(NDArray::ones({5}).isInfinite().any());
	EXPECT_FALSE(NDArray::ones({5}).isNaN().any());
}


// ============================================================
// New dtypes: F64, INT32, INT64
// ============================================================

TEST(NDArray_elementwise, F64_Basic) {
	NDArray a(ArrayList<double>({1.5, 2.5, 3.5}));
	EXPECT_EQ(a.type, F64);
	a.square();
	EXPECT_DOUBLE_EQ(a.get<double>({0}), 2.25);
	EXPECT_DOUBLE_EQ(a.get<double>({1}), 6.25);
}

TEST(NDArray_elementwise, INT32_INT64_Arithmetic) {
	NDArray a(ArrayList<int32_t>({10, 20, 30}));
	NDArray b(ArrayList<int32_t>({1, 2, 3}));
	NDArray c = a + b;
	EXPECT_EQ(c.type, INT32);
	EXPECT_EQ(c.get<int32_t>({0}), 11);
	EXPECT_EQ(c.get<int32_t>({2}), 33);

	NDArray d(ArrayList<int64_t>({(int64_t)1e12, (int64_t)2e12}));
	EXPECT_EQ(d.type, INT64);
	d.square();
	// 1e12^2 = 1e24 — may overflow int64; just check type stayed
	EXPECT_EQ(d.type, INT64);
}

TEST(NDArray_elementwise, INT32_plus_F32_PromotesToF64) {
	NDArray a(ArrayList<int32_t>({1, 2, 3}));
	NDArray b(ArrayList({0.5f, 0.5f, 0.5f}));
	NDArray c = a + b;
	EXPECT_EQ(c.type, F64);
	EXPECT_DOUBLE_EQ(c.get<double>({0}), 1.5);
	EXPECT_DOUBLE_EQ(c.get<double>({2}), 3.5);
}
