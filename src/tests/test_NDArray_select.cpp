
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
// Factories
// ============================================================

TEST(NDArray_select, Zeros_Ones_Full) {
	NDArray z = NDArray::zeros({3}, F32);
	EXPECT_EQ(z.type, F32);
	EXPECT_FLOAT_EQ(z.get<float>({0}), 0.0f);
	EXPECT_FLOAT_EQ(z.get<float>({2}), 0.0f);

	NDArray o = NDArray::ones({3}, F32);
	EXPECT_FLOAT_EQ(o.get<float>({0}), 1.0f);
	EXPECT_FLOAT_EQ(o.get<float>({2}), 1.0f);

	NDArray f = NDArray::full({2, 2}, INT32, -1);
	EXPECT_EQ(f.type, INT32);
	EXPECT_EQ(f.get<int32_t>({0, 0}), -1);
	EXPECT_EQ(f.get<int32_t>({1, 1}), -1);
}

TEST(NDArray_select, FullLike_ZerosLike) {
	NDArray ref(ArrayList({1.0f, 2.0f, 3.0f}));
	NDArray m1 = NDArray::fullLike(ref, -1.0f);
	EXPECT_EQ(m1.type, F32);
	EXPECT_EQ(m1.shape.get(0), 3);
	EXPECT_FLOAT_EQ(m1.get<float>({0}), -1.0f);
	EXPECT_FLOAT_EQ(m1.get<float>({2}), -1.0f);

	NDArray z = NDArray::zerosLike(ref);
	EXPECT_FLOAT_EQ(z.get<float>({1}), 0.0f);

	NDArray ones = NDArray::onesLike(ref);
	EXPECT_FLOAT_EQ(ones.get<float>({1}), 1.0f);
}

TEST(NDArray_select, Full_BINARY) {
	NDArray b = NDArray::full({8}, BINARY, 1);
	for (int i = 0; i < 8; ++i)
		EXPECT_EQ(b.get<int>({i}), 1);
	NDArray z = NDArray::full({8}, BINARY, 0);
	for (int i = 0; i < 8; ++i)
		EXPECT_EQ(z.get<int>({i}), 0);
}


// ============================================================
// Out-of-place unaries
// ============================================================

TEST(NDArray_select, Squared_DoesNotMutate) {
	NDArray b(ArrayList({2.0f, 3.0f}));
	NDArray b2 = b.squared();
	EXPECT_FLOAT_EQ(b.get<float>({0}), 2.0f); // original intact
	EXPECT_FLOAT_EQ(b2.get<float>({0}), 4.0f);
	EXPECT_FLOAT_EQ(b2.get<float>({1}), 9.0f);

	NDArray b4 = b.squared().squared(); // each squared() copies
	EXPECT_FLOAT_EQ(b.get<float>({0}), 2.0f);
	EXPECT_FLOAT_EQ(b4.get<float>({0}), 16.0f);
	EXPECT_FLOAT_EQ(b4.get<float>({1}), 81.0f);
}

TEST(NDArray_select, Negated_Absolute) {
	NDArray a(ArrayList({-2.0f, 3.0f}));
	NDArray n = a.negated();
	EXPECT_FLOAT_EQ(a.get<float>({0}), -2.0f);
	EXPECT_FLOAT_EQ(n.get<float>({0}), 2.0f);
	EXPECT_FLOAT_EQ(n.get<float>({1}), -3.0f);

	NDArray ab = a.absolute();
	EXPECT_FLOAT_EQ(a.get<float>({0}), -2.0f);
	EXPECT_FLOAT_EQ(ab.get<float>({0}), 2.0f);
	EXPECT_FLOAT_EQ(ab.get<float>({1}), 3.0f);
}


// ============================================================
// Element-wise comparison operators + ternary-style select
// ============================================================

TEST(NDArray_select, RelationalOperators_Elementwise) {
	NDArray a(ArrayList({1.0f, 5.0f, 3.0f}));
	NDArray b(ArrayList({2.0f, 5.0f, 1.0f}));

	NDArray gt = a > b;
	EXPECT_EQ(gt.type, BINARY);
	EXPECT_EQ(gt.get<int>({0}), 0);
	EXPECT_EQ(gt.get<int>({1}), 0);
	EXPECT_EQ(gt.get<int>({2}), 1);

	NDArray le = a <= b;
	EXPECT_EQ(le.get<int>({0}), 1);
	EXPECT_EQ(le.get<int>({1}), 1);
	EXPECT_EQ(le.get<int>({2}), 0);

	// Whole-array == still bool
	EXPECT_FALSE(a == b);
}

TEST(NDArray_select, TernaryStyle_Select_Choose) {
	// Desired: return a > b ? vb : va   (C++ cannot overload ?: )
	// Written as:  select(a > b, vb, va)  or  (a > b).choose(vb, va)
	NDArray a(ArrayList({1.0f, 5.0f, 3.0f}));
	NDArray b(ArrayList({2.0f, 4.0f, 3.0f}));
	NDArray va(ArrayList({10.0f, 20.0f, 30.0f}));
	NDArray vb(ArrayList({100.0f, 200.0f, 300.0f}));

	NDArray r1 = NDArray::select(a > b, vb, va);
	EXPECT_FLOAT_EQ(r1.get<float>({0}), 10.0f);  // 1>2 false → va
	EXPECT_FLOAT_EQ(r1.get<float>({1}), 200.0f); // 5>4 true → vb
	EXPECT_FLOAT_EQ(r1.get<float>({2}), 30.0f);  // 3>3 false → va

	NDArray r2 = (a > b).choose(vb, va);
	EXPECT_TRUE(r1 == r2);

	NDArray r3 = NDArray::where(a > b, vb, va);
	EXPECT_TRUE(r1 == r3);
}

TEST(NDArray_select, LeftScalar_Comparison) {
	NDArray a(ArrayList({1.0f, 5.0f, 3.0f}));
	NDArray m = 3.0f > a; // equivalent to a < 3
	EXPECT_EQ(m.get<int>({0}), 1);
	EXPECT_EQ(m.get<int>({1}), 0);
	EXPECT_EQ(m.get<int>({2}), 0);
}


// ============================================================
// divWhere
// ============================================================

TEST(NDArray_select, SafeDiv_ScalarWhenZero) {
	NDArray num(ArrayList({10.0f, 20.0f, 30.0f}));
	NDArray den(ArrayList({2.0f, 0.0f, 5.0f}));

	NDArray r = num.safeDiv(den, -1.0f);
	EXPECT_FLOAT_EQ(r.get<float>({0}), 5.0f);
	EXPECT_FLOAT_EQ(r.get<float>({1}), -1.0f); // zero den
	EXPECT_FLOAT_EQ(r.get<float>({2}), 6.0f);
}

TEST(NDArray_select, SafeDiv_ArrayWhenZero) {
	NDArray num(ArrayList({8.0f, 9.0f}));
	NDArray den(ArrayList({2.0f, 0.0f}));
	NDArray alt(ArrayList({100.0f, 200.0f}));

	NDArray r = num.safeDiv(den, alt);
	EXPECT_FLOAT_EQ(r.get<float>({0}), 4.0f);
	EXPECT_FLOAT_EQ(r.get<float>({1}), 200.0f);
}

TEST(NDArray_select, SafeDiv_Integer) {
	NDArray num(ArrayList<int32_t>({10, 20, 30}));
	NDArray den(ArrayList<int32_t>({2, 0, 5}));
	NDArray r = num.safeDiv(den, -1);
	EXPECT_EQ(r.type, INT32);
	EXPECT_EQ(r.get<int32_t>({0}), 5);
	EXPECT_EQ(r.get<int32_t>({1}), -1);
	EXPECT_EQ(r.get<int32_t>({2}), 6);
}

TEST(NDArray_select, Select_ScalarArms) {
	NDArray mask(ArrayList({1.0f, 0.0f, 1.0f}));
	NDArray r = NDArray::select(mask, 10.0, -1);
	EXPECT_EQ(r.type, F64);
	EXPECT_DOUBLE_EQ(r.get<double>({0}), 10.0);
	EXPECT_DOUBLE_EQ(r.get<double>({1}), -1.0);
	EXPECT_DOUBLE_EQ(r.get<double>({2}), 10.0);
}

TEST(NDArray_select, Choose_ScalarArms) {
	NDArray a(ArrayList({1.0f, 5.0f}));
	NDArray b(ArrayList({2.0f, 3.0f}));
	NDArray r = (a > b).choose(100.0f, 0.0f);
	EXPECT_FLOAT_EQ(r.get<float>({0}), 0.0f);
	EXPECT_FLOAT_EQ(r.get<float>({1}), 100.0f);
}


// ============================================================
// piecewise
// ============================================================

TEST(NDArray_select, Piecewise_FirstMaskWins) {
	NDArray m0(ArrayList({1.0f, 1.0f, 0.0f}));
	NDArray m1(ArrayList({1.0f, 0.0f, 1.0f})); // first index both true
	NDArray v0(ArrayList({10.0f, 20.0f, 30.0f}));
	NDArray v1(ArrayList({100.0f, 200.0f, 300.0f}));
	NDArray o(ArrayList({-1.0f, -1.0f, -1.0f}));

	NDArray r = NDArray::piecewise(m0, v0, m1, v1, o);
	EXPECT_FLOAT_EQ(r.get<float>({0}), 10.0f);  // m0 wins over m1
	EXPECT_FLOAT_EQ(r.get<float>({1}), 20.0f);  // m0
	EXPECT_FLOAT_EQ(r.get<float>({2}), 300.0f); // m1
}

TEST(NDArray_select, Piecewise_ScalarOtherwise_NoFullArrayNeeded) {
	// Scalar -1.0 is only written where selected; no fullLike required.
	NDArray m0(ArrayList({1.0f, 0.0f, 0.0f}));
	NDArray v0(ArrayList({10.0f, 20.0f, 30.0f}));
	NDArray r = NDArray::piecewise(m0, v0, -1.0);
	EXPECT_EQ(r.type, F64); // double otherwise promotes
	EXPECT_DOUBLE_EQ(r.get<double>({0}), 10.0);
	EXPECT_DOUBLE_EQ(r.get<double>({1}), -1.0);
	EXPECT_DOUBLE_EQ(r.get<double>({2}), -1.0);
}

TEST(NDArray_select, Piecewise_TruthyMaskAndScalar) {
	// piecewise(vb==0, when_vb0, b, main, -1.0) pattern
	NDArray b(ArrayList({2.0f, 0.0f, 3.0f}));
	NDArray main(ArrayList({7.0f, 8.0f, 9.0f}));
	NDArray when0(ArrayList({1.0f, 1.0f, 1.0f}));
	NDArray vb(ArrayList({0.0f, 1.0f, 1.0f}));

	NDArray r = NDArray::piecewise(vb == 0.0, when0, b, main, -1.0);
	EXPECT_DOUBLE_EQ(r.get<double>({0}), 1.0);  // vb==0
	EXPECT_DOUBLE_EQ(r.get<double>({1}), -1.0); // b==0 → otherwise
	EXPECT_DOUBLE_EQ(r.get<double>({2}), 9.0);  // b truthy → main
}


// ============================================================
// End-to-end sample fn (zero-denominator branches)
// ============================================================

/**
 * Vectorized form of:
 *   if (vb == 0) return va / (b*b);
 *   if (b == 0) return -1;
 *   return (a*a*vb + b*b*va) / (b*b*b*b);
 *
 * piecewise(vb==0, when_vb0, b, main, -1):
 *   if vb==0 → when_vb0; else if b truthy → main; else → -1
 *   (i.e. b==0 → -1 without a second equality mask)
 */
static NDArray sampleFn(const NDArray& a, const NDArray& b,
                        const NDArray& va, const NDArray& vb) {
	NDArray b2 = b.squared();
	NDArray b4 = b2.squared();
	NDArray when_vb0 = va.safeDiv(b2, 0.0);
	NDArray main = (a.squared() * vb + b2 * va).safeDiv(b4, 0.0);

	return NDArray::piecewise(
		vb == 0.0, when_vb0,
		b, main,
		-1.0);
}

TEST(NDArray_select, SampleFn_NormalBranch) {
	// a=2, b=2, va=3, vb=4
	// (4*4 + 4*3) / 16 = (16+12)/16 = 1.75
	NDArray a(ArrayList({2.0f}));
	NDArray b(ArrayList({2.0f}));
	NDArray va(ArrayList({3.0f}));
	NDArray vb(ArrayList({4.0f}));
	NDArray r = sampleFn(a, b, va, vb);
	EXPECT_FLOAT_EQ(r.get<float>({0}), 1.75f);
	// inputs not destroyed
	EXPECT_FLOAT_EQ(a.get<float>({0}), 2.0f);
	EXPECT_FLOAT_EQ(b.get<float>({0}), 2.0f);
}

TEST(NDArray_select, SampleFn_VbZero) {
	// vb=0, b=2 → va/(b*b) = 8/4 = 2
	NDArray a(ArrayList({1.0f}));
	NDArray b(ArrayList({2.0f}));
	NDArray va(ArrayList({8.0f}));
	NDArray vb(ArrayList({0.0f}));
	NDArray r = sampleFn(a, b, va, vb);
	EXPECT_FLOAT_EQ(r.get<float>({0}), 2.0f);
}

TEST(NDArray_select, SampleFn_BZero) {
	// b=0, vb=1 → -1
	NDArray a(ArrayList({1.0f}));
	NDArray b(ArrayList({0.0f}));
	NDArray va(ArrayList({8.0f}));
	NDArray vb(ArrayList({1.0f}));
	NDArray r = sampleFn(a, b, va, vb);
	EXPECT_FLOAT_EQ(r.get<float>({0}), -1.0f);
}

TEST(NDArray_select, SampleFn_MixedVector) {
	// three elements: normal, vb=0, b=0
	NDArray a(ArrayList({2.0f, 1.0f, 1.0f}));
	NDArray b(ArrayList({2.0f, 2.0f, 0.0f}));
	NDArray va(ArrayList({3.0f, 8.0f, 8.0f}));
	NDArray vb(ArrayList({4.0f, 0.0f, 1.0f}));
	NDArray r = sampleFn(a, b, va, vb);
	EXPECT_FLOAT_EQ(r.get<float>({0}), 1.75f);
	EXPECT_FLOAT_EQ(r.get<float>({1}), 2.0f);
	EXPECT_FLOAT_EQ(r.get<float>({2}), -1.0f);
}

TEST(NDArray_select, TernaryHelper_a_gt_b) {
	// Review example: (a, b, va, vb) → a > b ? vb : va
	auto crude = [](const NDArray& a, const NDArray& b,
	                const NDArray& va, const NDArray& vb) {
		return NDArray::select(a > b, vb, va);
	};

	NDArray a(ArrayList({1.0f, 9.0f}));
	NDArray b(ArrayList({5.0f, 3.0f}));
	NDArray va(ArrayList({10.0f, 20.0f}));
	NDArray vb(ArrayList({100.0f, 200.0f}));
	NDArray r = crude(a, b, va, vb);
	EXPECT_FLOAT_EQ(r.get<float>({0}), 10.0f);
	EXPECT_FLOAT_EQ(r.get<float>({1}), 200.0f);
}
