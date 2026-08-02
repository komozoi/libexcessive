
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
// Same-type element-wise arithmetic
// ============================================================

TEST(NDArray_arithmetic, F32_Add) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f}));
	NDArray b(ArrayList({4.0f, 5.0f, 6.0f}));

	NDArray c = a + b;
	EXPECT_EQ(c.type, F32);
	EXPECT_FLOAT_EQ(c.get<float>({0}), 5.0f);
	EXPECT_FLOAT_EQ(c.get<float>({1}), 7.0f);
	EXPECT_FLOAT_EQ(c.get<float>({2}), 9.0f);

	// operands unchanged
	EXPECT_FLOAT_EQ(a.get<float>({0}), 1.0f);
	EXPECT_FLOAT_EQ(b.get<float>({0}), 4.0f);
}

TEST(NDArray_arithmetic, F32_SubMulDiv) {
	NDArray a(ArrayList({10.0f, 20.0f}));
	NDArray b(ArrayList({3.0f, 5.0f}));

	NDArray sub = a - b;
	EXPECT_FLOAT_EQ(sub.get<float>({0}), 7.0f);
	EXPECT_FLOAT_EQ(sub.get<float>({1}), 15.0f);

	NDArray mul = a * b;
	EXPECT_FLOAT_EQ(mul.get<float>({0}), 30.0f);
	EXPECT_FLOAT_EQ(mul.get<float>({1}), 100.0f);

	NDArray div = a / b;
	EXPECT_FLOAT_EQ(div.get<float>({0}), 10.0f / 3.0f);
	EXPECT_FLOAT_EQ(div.get<float>({1}), 4.0f);
}

TEST(NDArray_arithmetic, UINT8_Add) {
	NDArray a(ArrayList<uint8_t>({1, 2, 3}));
	NDArray b(ArrayList<uint8_t>({4, 5, 6}));

	NDArray c = a + b;
	EXPECT_EQ(c.type, UINT8);
	EXPECT_EQ(c.get<uint8_t>({0}), 5);
	EXPECT_EQ(c.get<uint8_t>({1}), 7);
	EXPECT_EQ(c.get<uint8_t>({2}), 9);
}

TEST(NDArray_arithmetic, UINT256_AddMul) {
	NDArray a({2}, UINT256);
	NDArray b({2}, UINT256);
	a.set({0}, 10);
	a.set({1}, 20);
	b.set({0}, 3);
	b.set({1}, 4);

	NDArray sum = a + b;
	EXPECT_EQ(sum.type, UINT256);
	EXPECT_EQ(sum.get<uint256_t>({0}), uint256_t(13));
	EXPECT_EQ(sum.get<uint256_t>({1}), uint256_t(24));

	NDArray prod = a * b;
	EXPECT_EQ(prod.get<uint256_t>({0}), uint256_t(30));
	EXPECT_EQ(prod.get<uint256_t>({1}), uint256_t(80));
}


// ============================================================
// Type promotion — never lose precision unless convert() was used
// ============================================================

TEST(NDArray_arithmetic, Promote_UINT8_plus_F32_to_F32) {
	NDArray a(ArrayList<uint8_t>({1, 2, 3}));
	NDArray b(ArrayList({0.5f, 0.5f, 0.5f}));

	NDArray c = a + b;
	EXPECT_EQ(c.type, F32);
	EXPECT_FLOAT_EQ(c.get<float>({0}), 1.5f);
	EXPECT_FLOAT_EQ(c.get<float>({1}), 2.5f);
	EXPECT_FLOAT_EQ(c.get<float>({2}), 3.5f);
}

TEST(NDArray_arithmetic, Promote_UINT8_plus_UINT256_to_UINT256) {
	NDArray a(ArrayList<uint8_t>({1, 2}));
	NDArray b({2}, UINT256);
	b.set({0}, 100);
	b.set({1}, 200);

	NDArray c = a + b;
	EXPECT_EQ(c.type, UINT256);
	EXPECT_EQ(c.get<uint256_t>({0}), uint256_t(101));
	EXPECT_EQ(c.get<uint256_t>({1}), uint256_t(202));
}

TEST(NDArray_arithmetic, Promote_BINARY_plus_BINARY_to_UINT8) {
	// 1+1 must not stay in BINARY (would lose information)
	NDArray a({3}, BINARY);
	NDArray b({3}, BINARY);
	a.set({0}, 1);
	a.set({1}, 1);
	a.set({2}, 0);
	b.set({0}, 1);
	b.set({1}, 0);
	b.set({2}, 1);

	NDArray c = a + b;
	EXPECT_EQ(c.type, UINT8);
	EXPECT_EQ(c.get<uint8_t>({0}), 2);
	EXPECT_EQ(c.get<uint8_t>({1}), 1);
	EXPECT_EQ(c.get<uint8_t>({2}), 1);
}

TEST(NDArray_arithmetic, Promote_BINARY_plus_UINT8_to_UINT8) {
	NDArray a({2}, BINARY);
	a.set({0}, 1);
	a.set({1}, 0);
	NDArray b(ArrayList<uint8_t>({5, 7}));

	NDArray c = a + b;
	EXPECT_EQ(c.type, UINT8);
	EXPECT_EQ(c.get<uint8_t>({0}), 6);
	EXPECT_EQ(c.get<uint8_t>({1}), 7);
}

TEST(NDArray_arithmetic, Promote_BINARY_plus_F32_to_F32) {
	NDArray a({2}, BINARY);
	a.set({0}, 1);
	a.set({1}, 0);
	NDArray b(ArrayList({2.5f, 3.5f}));

	NDArray c = a + b;
	EXPECT_EQ(c.type, F32);
	EXPECT_FLOAT_EQ(c.get<float>({0}), 3.5f);
	EXPECT_FLOAT_EQ(c.get<float>({1}), 3.5f);
}

TEST(NDArray_arithmetic, NoCommonType_F32_and_UINT256_Throws) {
	// F32 is signed / fractional; UINT256 is wide unsigned — no lossless common type
	NDArray a(ArrayList({1.0f, 2.0f}));
	NDArray b({2}, UINT256);
	b.set({0}, 1);
	b.set({1}, 2);

	EXPECT_THROW(a + b, std::invalid_argument);
	EXPECT_THROW(a * b, std::invalid_argument);
}


// ============================================================
// Scalar arithmetic + promotion
// ============================================================

TEST(NDArray_arithmetic, F32_plus_FloatScalar) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f}));
	NDArray c = a + 10.0f;
	EXPECT_EQ(c.type, F32);
	EXPECT_FLOAT_EQ(c.get<float>({0}), 11.0f);
	EXPECT_FLOAT_EQ(c.get<float>({2}), 13.0f);
}

TEST(NDArray_arithmetic, UINT8_plus_FloatScalar_PromotesToF32) {
	NDArray a(ArrayList<uint8_t>({1, 2, 3}));
	NDArray c = a + 0.5f;
	EXPECT_EQ(c.type, F32);
	EXPECT_FLOAT_EQ(c.get<float>({0}), 1.5f);
	EXPECT_FLOAT_EQ(c.get<float>({1}), 2.5f);
	EXPECT_FLOAT_EQ(c.get<float>({2}), 3.5f);
}

TEST(NDArray_arithmetic, UINT8_plus_IntScalar_PromotesToF32) {
	// int is signed; only signed storage type is F32
	NDArray a(ArrayList<uint8_t>({1, 2, 3}));
	NDArray c = a + 10;
	EXPECT_EQ(c.type, F32);
	EXPECT_FLOAT_EQ(c.get<float>({0}), 11.0f);
	EXPECT_FLOAT_EQ(c.get<float>({2}), 13.0f);
}

TEST(NDArray_arithmetic, UINT256_plus_IntScalar_StaysUINT256) {
	NDArray a({2}, UINT256);
	a.set({0}, 5);
	a.set({1}, 6);
	NDArray c = a + 10;
	EXPECT_EQ(c.type, UINT256);
	EXPECT_EQ(c.get<uint256_t>({0}), uint256_t(15));
	EXPECT_EQ(c.get<uint256_t>({1}), uint256_t(16));
}

TEST(NDArray_arithmetic, Scalar_MulDivSub) {
	NDArray a(ArrayList({8.0f, 6.0f}));

	NDArray m = a * 2.0f;
	EXPECT_FLOAT_EQ(m.get<float>({0}), 16.0f);
	EXPECT_FLOAT_EQ(m.get<float>({1}), 12.0f);

	NDArray d = a / 2.0f;
	EXPECT_FLOAT_EQ(d.get<float>({0}), 4.0f);
	EXPECT_FLOAT_EQ(d.get<float>({1}), 3.0f);

	NDArray s = a - 1.0f;
	EXPECT_FLOAT_EQ(s.get<float>({0}), 7.0f);
	EXPECT_FLOAT_EQ(s.get<float>({1}), 5.0f);
}

TEST(NDArray_arithmetic, DoubleScalar_TreatedAsFloat) {
	// F64 not yet a storage type; double promotes like F32
	NDArray a(ArrayList<uint8_t>({2, 4}));
	NDArray c = a * 1.5;
	EXPECT_EQ(c.type, F32);
	EXPECT_FLOAT_EQ(c.get<float>({0}), 3.0f);
	EXPECT_FLOAT_EQ(c.get<float>({1}), 6.0f);
}


// ============================================================
// Compound assignment (in-place; keeps destination type)
// ============================================================

TEST(NDArray_arithmetic, CompoundAdd_F32) {
	NDArray a(ArrayList({1.0f, 2.0f}));
	a += 5.0f;
	EXPECT_FLOAT_EQ(a.get<float>({0}), 6.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1}), 7.0f);

	NDArray b(ArrayList({1.0f, 1.0f}));
	a += b;
	EXPECT_FLOAT_EQ(a.get<float>({0}), 7.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1}), 8.0f);
}

TEST(NDArray_arithmetic, CompoundMul_UINT8_IntPromotesToF32) {
	// int is signed → promote destination to F32
	NDArray a(ArrayList<uint8_t>({2, 3, 4}));
	a *= 2;
	EXPECT_EQ(a.type, F32);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 4.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1}), 6.0f);
	EXPECT_FLOAT_EQ(a.get<float>({2}), 8.0f);
}

TEST(NDArray_arithmetic, Compound_PromotesDestination_UINT8_plus_F32) {
	NDArray a(ArrayList<uint8_t>({1, 2}));
	NDArray b(ArrayList({0.5f, 1.5f}));
	a += b;
	EXPECT_EQ(a.type, F32);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 1.5f);
	EXPECT_FLOAT_EQ(a.get<float>({1}), 3.5f);
}

TEST(NDArray_arithmetic, Compound_F32_plus_UINT8_StaysF32) {
	// Destination already holds the promoted type
	NDArray a(ArrayList({1.0f, 2.0f}));
	NDArray b(ArrayList<uint8_t>({3, 4}));
	a += b;
	EXPECT_EQ(a.type, F32);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 4.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1}), 6.0f);
}

TEST(NDArray_arithmetic, Compound_BINARY_plus_BINARY_PromotesToUINT8) {
	NDArray a({2}, BINARY);
	NDArray b({2}, BINARY);
	a.set({0}, 1);
	a.set({1}, 1);
	b.set({0}, 1);
	b.set({1}, 0);
	a += b;
	EXPECT_EQ(a.type, UINT8);
	EXPECT_EQ(a.get<uint8_t>({0}), 2);
	EXPECT_EQ(a.get<uint8_t>({1}), 1);
}

TEST(NDArray_arithmetic, Compound_UINT8_plus_FloatScalar_PromotesToF32) {
	NDArray a(ArrayList<uint8_t>({1, 2, 3}));
	a += 0.5f;
	EXPECT_EQ(a.type, F32);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 1.5f);
	EXPECT_FLOAT_EQ(a.get<float>({1}), 2.5f);
	EXPECT_FLOAT_EQ(a.get<float>({2}), 3.5f);
}

TEST(NDArray_arithmetic, Compound_NoCommonType_Throws) {
	NDArray a(ArrayList({1.0f, 2.0f}));
	NDArray b({2}, UINT256);
	b.set({0}, 1);
	b.set({1}, 2);
	EXPECT_THROW(a += b, std::invalid_argument);
}


// ============================================================
// Named in-place methods: add / sub / mul / div
// ============================================================

TEST(NDArray_arithmetic, NamedAdd_SameType) {
	NDArray a(ArrayList({1.0f, 2.0f}));
	NDArray b(ArrayList({3.0f, 4.0f}));
	a.add(b);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 4.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1}), 6.0f);
}

TEST(NDArray_arithmetic, NamedAdd_PromotesDestination) {
	NDArray a(ArrayList<uint8_t>({1, 2}));
	NDArray b(ArrayList({3.0f, 4.0f}));
	a.add(b);
	EXPECT_EQ(a.type, F32);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 4.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1}), 6.0f);
	// other is left unchanged
	EXPECT_EQ(b.type, F32);
	EXPECT_FLOAT_EQ(b.get<float>({0}), 3.0f);
}

TEST(NDArray_arithmetic, NamedSubMulDiv) {
	NDArray a(ArrayList({10.0f, 20.0f}));
	NDArray b(ArrayList({2.0f, 4.0f}));

	a.sub(b);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 8.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1}), 16.0f);

	a.mul(b);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 16.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1}), 64.0f);

	a.div(b);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 8.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1}), 16.0f);
}


// ============================================================
// Broadcast ops — other.shape is a prefix of this->shape
// ============================================================

TEST(NDArray_arithmetic, BroadcastAdd_Prefix) {
	// this: [2, 3], other: [2]  (prefix)
	NDArray a({2, 3}, ArrayList({
		1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f
	}));
	NDArray b(ArrayList({10.0f, 20.0f}));

	a.broadcastAdd(b);

	EXPECT_FLOAT_EQ(a.get<float>({0, 0}), 11.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0, 1}), 11.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0, 2}), 11.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 0}), 21.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 1}), 21.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 2}), 21.0f);
}

TEST(NDArray_arithmetic, BroadcastSub_Prefix) {
	NDArray a({2, 2}, ArrayList({
		10.0f, 10.0f,
		20.0f, 20.0f
	}));
	NDArray b(ArrayList({3.0f, 5.0f}));

	a.broadcastSub(b);

	EXPECT_FLOAT_EQ(a.get<float>({0, 0}), 7.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0, 1}), 7.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 0}), 15.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 1}), 15.0f);
}

TEST(NDArray_arithmetic, BroadcastMul_Prefix) {
	NDArray a({2, 3}, ArrayList({
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f
	}));
	NDArray b(ArrayList({10.0f, 100.0f}));

	a.broadcastMul(b);

	EXPECT_FLOAT_EQ(a.get<float>({0, 0}), 10.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0, 1}), 20.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0, 2}), 30.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 0}), 400.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 1}), 500.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 2}), 600.0f);
}

TEST(NDArray_arithmetic, BroadcastDiv_Prefix) {
	NDArray a({2, 2}, ArrayList({
		10.0f, 20.0f,
		30.0f, 40.0f
	}));
	NDArray b(ArrayList({2.0f, 10.0f}));

	a.broadcastDiv(b);

	EXPECT_FLOAT_EQ(a.get<float>({0, 0}), 5.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0, 1}), 10.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 0}), 3.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 1}), 4.0f);
}

TEST(NDArray_arithmetic, BroadcastAdd_FullPrefix) {
	// other shape equals this shape → same as element-wise
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f}));
	NDArray b(ArrayList({4.0f, 5.0f, 6.0f}));
	a.broadcastAdd(b);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 5.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1}), 7.0f);
	EXPECT_FLOAT_EQ(a.get<float>({2}), 9.0f);
}

TEST(NDArray_arithmetic, Broadcast_NotPrefix_Throws) {
	NDArray a({2, 3}, F32);
	NDArray b({3}, F32); // not a prefix of [2,3]
	EXPECT_THROW(a.broadcastAdd(b), std::invalid_argument);
	EXPECT_THROW(a.broadcastSub(b), std::invalid_argument);
	EXPECT_THROW(a.broadcastMul(b), std::invalid_argument);
	EXPECT_THROW(a.broadcastDiv(b), std::invalid_argument);
}

TEST(NDArray_arithmetic, Broadcast_PromotesDestination) {
	// UINT8 destination + F32 broadcast source → promote to F32
	NDArray a({2, 2}, ArrayList<uint8_t>({1, 1, 1, 1}));
	NDArray b(ArrayList({0.5f, 2.0f}));

	a.broadcastAdd(b);
	EXPECT_EQ(a.type, F32);
	EXPECT_FLOAT_EQ(a.get<float>({0, 0}), 1.5f);
	EXPECT_FLOAT_EQ(a.get<float>({0, 1}), 1.5f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 0}), 3.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 1}), 3.0f);
}

TEST(NDArray_arithmetic, Broadcast_ScalarOther) {
	// other with empty shape (scalar) is a prefix of everything
	NDArray a({2, 2}, ArrayList({1.0f, 2.0f, 3.0f, 4.0f}));
	NDArray s(F32, 10.0f);

	a.broadcastMul(s);
	EXPECT_FLOAT_EQ(a.get<float>({0, 0}), 10.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0, 1}), 20.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 0}), 30.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 1}), 40.0f);
}


// ============================================================
// Shape validation
// ============================================================

TEST(NDArray_arithmetic, ShapeMismatch_Throws) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f}));
	NDArray b(ArrayList({1.0f, 2.0f}));
	EXPECT_THROW(a + b, std::invalid_argument);
	EXPECT_THROW(a += b, std::invalid_argument);
}

TEST(NDArray_arithmetic, Matrix_Elementwise) {
	NDArray a({2, 2}, ArrayList({1.0f, 2.0f, 3.0f, 4.0f}));
	NDArray b({2, 2}, ArrayList({10.0f, 20.0f, 30.0f, 40.0f}));

	NDArray c = a + b;
	EXPECT_FLOAT_EQ(c.get<float>({0, 0}), 11.0f);
	EXPECT_FLOAT_EQ(c.get<float>({0, 1}), 22.0f);
	EXPECT_FLOAT_EQ(c.get<float>({1, 0}), 33.0f);
	EXPECT_FLOAT_EQ(c.get<float>({1, 1}), 44.0f);
}

TEST(NDArray_arithmetic, ScalarArray_Ops) {
	NDArray a(F32, 5.0f);
	NDArray b(F32, 3.0f);

	NDArray c = a + b;
	EXPECT_EQ(c.shape.size(), 0);
	EXPECT_FLOAT_EQ(c.get<float>({}), 8.0f);

	NDArray d = a * 2.0f;
	EXPECT_FLOAT_EQ(d.get<float>({}), 10.0f);
}
