
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

TEST(NDArray_arithmetic, F16_Add_StaysF16) {
	NDArray a({3}, F16);
	NDArray b({3}, F16);
	a.set({0}, 1.0f);
	a.set({1}, 2.0f);
	a.set({2}, 0.5f);
	b.set({0}, 3.0f);
	b.set({1}, 4.0f);
	b.set({2}, 0.5f);
	NDArray c = a + b;
	EXPECT_EQ(c.type, F16);
	EXPECT_FLOAT_EQ(c.get<float>({0}), 4.0f);
	EXPECT_FLOAT_EQ(c.get<float>({1}), 6.0f);
	EXPECT_FLOAT_EQ(c.get<float>({2}), 1.0f);
	EXPECT_EQ(a.type, F16);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 1.0f);

	a += b;
	EXPECT_EQ(a.type, F16);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 4.0f);

	NDArray d = a + 1.0f;
	EXPECT_EQ(d.type, F16);
	EXPECT_FLOAT_EQ(d.get<float>({0}), 5.0f);
	a += 1;
	EXPECT_EQ(a.type, F16);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 5.0f);
}

TEST(NDArray_arithmetic, BF16_Add_StaysBF16) {
	NDArray a({2}, BF16);
	NDArray b({2}, BF16);
	a.set({0}, 1.5f);
	a.set({1}, -0.5f);
	b.set({0}, 2.5f);
	b.set({1}, 0.5f);
	NDArray c = a + b;
	EXPECT_EQ(c.type, BF16);
	EXPECT_FLOAT_EQ(c.get<float>({0}), 4.0f);
	EXPECT_FLOAT_EQ(c.get<float>({1}), 0.0f);
}

TEST(NDArray_arithmetic, BF16_Add_F32) {
	NDArray a({2}, BF16);
	a.set({0}, 1.5f);
	a.set({1}, -0.5f);
	NDArray b(ArrayList({2.5f, 0.5f}));
	NDArray c = a + b;
	EXPECT_EQ(c.type, F32);
	EXPECT_FLOAT_EQ(c.get<float>({0}), 4.0f);
	EXPECT_FLOAT_EQ(c.get<float>({1}), 0.0f);
}

TEST(NDArray_arithmetic, F16_UnaryAndCompare_StayHalf) {
	NDArray a({3}, F16);
	a.set({0}, -2.0f);
	a.set({1}, 0.0f);
	a.set({2}, 1.0f);
	NDArray n = a.negated();
	EXPECT_EQ(n.type, F16);
	EXPECT_FLOAT_EQ(n.get<float>({0}), 2.0f);
	EXPECT_FLOAT_EQ(n.get<float>({2}), -1.0f);

	NDArray s = a;
	s.sin();
	EXPECT_EQ(s.type, F16);

	NDArray b({3}, F16);
	b.set({0}, -2.0f);
	b.set({1}, 1.0f);
	b.set({2}, 1.0f);
	NDArray eq = a.equal(b);
	EXPECT_EQ(eq.type, BINARY);
	EXPECT_EQ(eq.get<int>({0}), 1);
	EXPECT_EQ(eq.get<int>({1}), 0);
	EXPECT_EQ(eq.get<int>({2}), 1);

	NDArray cmp = a.compare(b);
	EXPECT_EQ(cmp.type, INT3);
	EXPECT_EQ(cmp.get<int>({0}), 0);
	EXPECT_EQ(cmp.get<int>({1}), -1);

	NDArray m = a.minimum(b);
	EXPECT_EQ(m.type, F16);
	EXPECT_FLOAT_EQ(m.get<float>({1}), 0.0f);
	NDArray m1 = a.minimum(1.0f);
	EXPECT_EQ(m1.type, F16);
	EXPECT_FLOAT_EQ(m1.get<float>({0}), -2.0f);
	EXPECT_FLOAT_EQ(m1.get<float>({2}), 1.0f);
}

TEST(NDArray_arithmetic, INT8_Add_StaysINT8) {
	NDArray a({3}, INT8);
	NDArray b({3}, INT8);
	a.set({0}, -2);
	a.set({1}, 10);
	a.set({2}, 127);
	b.set({0}, -3);
	b.set({1}, 5);
	b.set({2}, 1);
	NDArray c = a + b;
	EXPECT_EQ(c.type, INT8);
	EXPECT_EQ(c.get<int>({0}), -5);
	EXPECT_EQ(c.get<int>({1}), 15);
	EXPECT_EQ(c.get<int8_t>({2}), (int8_t)-128);
}

TEST(NDArray_arithmetic, Promote_INT8_plus_INT32_to_INT32) {
	NDArray a({2}, INT8);
	NDArray b({2}, INT32);
	a.set({0}, -1);
	a.set({1}, 2);
	b.set({0}, 10);
	b.set({1}, 20);
	NDArray c = a + b;
	EXPECT_EQ(c.type, INT32);
	EXPECT_EQ(c.get<int32_t>({0}), 9);
	EXPECT_EQ(c.get<int32_t>({1}), 22);
}

TEST(NDArray_arithmetic, Promote_INT8_plus_F32_to_F32) {
	NDArray a({2}, INT8);
	a.set({0}, -1);
	a.set({1}, 3);
	NDArray b(ArrayList({0.5f, 0.5f}));
	NDArray c = a + b;
	EXPECT_EQ(c.type, F32);
	EXPECT_FLOAT_EQ(c.get<float>({0}), -0.5f);
	EXPECT_FLOAT_EQ(c.get<float>({1}), 3.5f);
}

TEST(NDArray_arithmetic, Promote_INT8_plus_UINT8_to_INT32) {
	NDArray a({2}, INT8);
	NDArray b({2}, UINT8);
	a.set({0}, -1);
	a.set({1}, 2);
	b.set({0}, 200);
	b.set({1}, 3);
	NDArray c = a + b;
	EXPECT_EQ(c.type, INT32);
	EXPECT_EQ(c.get<int32_t>({0}), 199);
	EXPECT_EQ(c.get<int32_t>({1}), 5);
}

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

TEST(NDArray_arithmetic, UINT8_plus_IntScalar_StaysUINT8) {
	NDArray a(ArrayList<uint8_t>({1, 2, 3}));
	NDArray c = a + 10;
	EXPECT_EQ(c.type, UINT8);
	EXPECT_EQ(c.get<uint8_t>({0}), 11);
	EXPECT_EQ(c.get<uint8_t>({2}), 13);
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

TEST(NDArray_arithmetic, F32_plus_IntScalar_StaysF32) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f}));
	NDArray c = a + 1;
	EXPECT_EQ(c.type, F32);
	EXPECT_FLOAT_EQ(c.get<float>({0}), 2.0f);
	EXPECT_FLOAT_EQ(c.get<float>({2}), 4.0f);
	NDArray d = a + 1.0f;
	EXPECT_EQ(d.type, F32);
	EXPECT_FLOAT_EQ(d.get<float>({0}), c.get<float>({0}));
	EXPECT_FLOAT_EQ(d.get<float>({2}), c.get<float>({2}));
}

TEST(NDArray_arithmetic, F32_plus_DoubleOne_StaysF32) {
	NDArray a(ArrayList({1.0f, 2.0f}));
	NDArray c = a + 1.0;
	EXPECT_EQ(c.type, F32);
	EXPECT_FLOAT_EQ(c.get<float>({0}), 2.0f);
	EXPECT_FLOAT_EQ(c.get<float>({1}), 3.0f);
}

TEST(NDArray_arithmetic, UINT8_plus_256_PromotesToINT32) {
	NDArray a(ArrayList<uint8_t>({1, 2, 255}));
	NDArray c = a + 256;
	EXPECT_EQ(c.type, INT32);
	EXPECT_EQ(c.get<int32_t>({0}), 257);
	EXPECT_EQ(c.get<int32_t>({2}), 511);
}

TEST(NDArray_arithmetic, UINT8_plus_Neg1_PromotesToINT32) {
	NDArray a(ArrayList<uint8_t>({1, 2}));
	NDArray c = a + (-1);
	EXPECT_EQ(c.type, INT32);
	EXPECT_EQ(c.get<int32_t>({0}), 0);
	EXPECT_EQ(c.get<int32_t>({1}), 1);
}

TEST(NDArray_arithmetic, UINT8_255_plus_1_Wraps) {
	NDArray a(ArrayList<uint8_t>({255, 1}));
	NDArray c = a + 1;
	EXPECT_EQ(c.type, UINT8);
	EXPECT_EQ(c.get<uint8_t>({0}), 0);
	EXPECT_EQ(c.get<uint8_t>({1}), 2);
}

TEST(NDArray_arithmetic, UINT8_plus_ExactFloat_StaysUINT8) {
	NDArray a(ArrayList<uint8_t>({1, 2, 3}));
	NDArray c = a + 1.0f;
	EXPECT_EQ(c.type, UINT8);
	EXPECT_EQ(c.get<uint8_t>({0}), 2);
	EXPECT_EQ(c.get<uint8_t>({2}), 4);
}

TEST(NDArray_arithmetic, INT8_plus_IntScalar_StaysINT8) {
	NDArray a({3}, INT8);
	a.set({0}, -2);
	a.set({1}, 10);
	a.set({2}, 126);
	NDArray c = a + 1;
	EXPECT_EQ(c.type, INT8);
	EXPECT_EQ(c.get<int>({0}), -1);
	EXPECT_EQ(c.get<int>({1}), 11);
	EXPECT_EQ(c.get<int8_t>({2}), (int8_t)127);
}

TEST(NDArray_arithmetic, INT8_plus_128_PromotesToINT32) {
	NDArray a({2}, INT8);
	a.set({0}, 1);
	a.set({1}, -2);
	NDArray c = a + 128;
	EXPECT_EQ(c.type, INT32);
	EXPECT_EQ(c.get<int32_t>({0}), 129);
	EXPECT_EQ(c.get<int32_t>({1}), 126);
}

TEST(NDArray_arithmetic, BINARY_plus_1_StaysBINARY) {
	NDArray a({3}, BINARY);
	a.set({0}, 0);
	a.set({1}, 1);
	a.set({2}, 1);
	NDArray c = a + 1;
	EXPECT_EQ(c.type, BINARY);
	EXPECT_EQ(c.get<int>({0}), 1);
	EXPECT_EQ(c.get<int>({1}), 0);
	EXPECT_EQ(c.get<int>({2}), 0);
}

TEST(NDArray_arithmetic, BINARY_plus_2_PromotesToINT32) {
	NDArray a({2}, BINARY);
	a.set({0}, 1);
	a.set({1}, 0);
	NDArray c = a + 2;
	EXPECT_EQ(c.type, INT32);
	EXPECT_EQ(c.get<int32_t>({0}), 3);
	EXPECT_EQ(c.get<int32_t>({1}), 2);
}

TEST(NDArray_arithmetic, UINT8_NamedScalar_StaysUINT8) {
	NDArray a(ArrayList<uint8_t>({2, 3, 4}));
	NDArray p = a.pow(2);
	EXPECT_EQ(p.type, UINT8);
	EXPECT_EQ(p.get<uint8_t>({0}), 4);
	EXPECT_EQ(p.get<uint8_t>({2}), 16);
	NDArray m = a.minimum(3);
	EXPECT_EQ(m.type, UINT8);
	EXPECT_EQ(m.get<uint8_t>({0}), 2);
	EXPECT_EQ(m.get<uint8_t>({2}), 3);
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

TEST(NDArray_arithmetic, DoubleScalar_PromotesToF64) {
	NDArray a(ArrayList<uint8_t>({2, 4}));
	NDArray c = a * 1.5;
	EXPECT_EQ(c.type, F64);
	EXPECT_DOUBLE_EQ(c.get<double>({0}), 3.0);
	EXPECT_DOUBLE_EQ(c.get<double>({1}), 6.0);
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

TEST(NDArray_arithmetic, CompoundMul_UINT8_IntStaysUINT8) {
	NDArray a(ArrayList<uint8_t>({2, 3, 4}));
	a *= 2;
	EXPECT_EQ(a.type, UINT8);
	EXPECT_EQ(a.get<uint8_t>({0}), 4);
	EXPECT_EQ(a.get<uint8_t>({1}), 6);
	EXPECT_EQ(a.get<uint8_t>({2}), 8);
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
// Broadcast ops — NumPy: right-align, size-1 axes expand
// ============================================================

TEST(NDArray_arithmetic, BroadcastAdd_Trailing) {
	// {3} → {2,3}: the case views accept and prefix broadcast rejected
	NDArray a({2, 3}, ArrayList({
		1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f
	}));
	NDArray b(ArrayList({10.0f, 20.0f, 30.0f}));

	ASSERT_TRUE(b.isBroadcastableTo(a.shape));
	a.broadcastAdd(b);

	EXPECT_FLOAT_EQ(a.get<float>({0, 0}), 11.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0, 1}), 21.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0, 2}), 31.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 0}), 11.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 1}), 21.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 2}), 31.0f);
}

TEST(NDArray_arithmetic, BroadcastAdd_Size1) {
	// {2,1} → {2,3}
	NDArray a({2, 3}, ArrayList({
		1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f
	}));
	NDArray b({2, 1}, ArrayList({10.0f, 20.0f}));

	a.broadcastAdd(b);

	EXPECT_FLOAT_EQ(a.get<float>({0, 0}), 11.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0, 1}), 11.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0, 2}), 11.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 0}), 21.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 1}), 21.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 2}), 21.0f);
}

TEST(NDArray_arithmetic, BroadcastSub_Trailing) {
	NDArray a({2, 2}, ArrayList({
		10.0f, 10.0f,
		20.0f, 20.0f
	}));
	NDArray b(ArrayList({3.0f, 5.0f}));

	a.broadcastSub(b);

	EXPECT_FLOAT_EQ(a.get<float>({0, 0}), 7.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0, 1}), 5.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 0}), 17.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 1}), 15.0f);
}

TEST(NDArray_arithmetic, BroadcastMul_Size1) {
	NDArray a({2, 3}, ArrayList({
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f
	}));
	NDArray b({2, 1}, ArrayList({10.0f, 100.0f}));

	a.broadcastMul(b);

	EXPECT_FLOAT_EQ(a.get<float>({0, 0}), 10.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0, 1}), 20.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0, 2}), 30.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 0}), 400.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 1}), 500.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 2}), 600.0f);
}

TEST(NDArray_arithmetic, BroadcastDiv_Trailing) {
	NDArray a({2, 2}, ArrayList({
		10.0f, 20.0f,
		30.0f, 40.0f
	}));
	NDArray b(ArrayList({2.0f, 10.0f}));

	a.broadcastDiv(b);

	EXPECT_FLOAT_EQ(a.get<float>({0, 0}), 5.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0, 1}), 2.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 0}), 15.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 1}), 4.0f);
}

TEST(NDArray_arithmetic, BroadcastAdd_InsertMiddle) {
	NDArray a({2, 3, 4}, F32);
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 3; ++j)
			for (int k = 0; k < 4; ++k)
				a.set({i, j, k}, 1.0f);
	NDArray b({2, 3, 1}, F32);
	float v = 0.0f;
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 3; ++j)
			b.set({i, j, 0}, v++);

	a.broadcastAdd(b);
	EXPECT_FLOAT_EQ(a.get<float>({0, 0, 0}), 1.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0, 0, 3}), 1.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 2, 1}), 6.0f);
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

TEST(NDArray_arithmetic, Broadcast_NotBroadcastable_Throws) {
	NDArray a({2, 3}, F32);
	NDArray b({4}, F32);
	EXPECT_THROW(a.broadcastAdd(b), std::invalid_argument);
	EXPECT_THROW(a.broadcastSub(b), std::invalid_argument);
	EXPECT_THROW(a.broadcastMul(b), std::invalid_argument);
	EXPECT_THROW(a.broadcastDiv(b), std::invalid_argument);

	NDArray lead({2}, F32); // {2} vs last axis 3
	EXPECT_FALSE(lead.isBroadcastableTo(a.shape));
	EXPECT_THROW(a.broadcastAdd(lead), std::invalid_argument);
}

TEST(NDArray_arithmetic, Broadcast_PromotesDestination) {
	// UINT8 destination + F32 trailing broadcast → promote to F32
	NDArray a({2, 2}, ArrayList<uint8_t>({1, 1, 1, 1}));
	NDArray b(ArrayList({0.5f, 2.0f}));

	a.broadcastAdd(b);
	EXPECT_EQ(a.type, F32);
	EXPECT_FLOAT_EQ(a.get<float>({0, 0}), 1.5f);
	EXPECT_FLOAT_EQ(a.get<float>({0, 1}), 3.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 0}), 1.5f);
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


// ============================================================
// Left-hand scalar operators:  scalar ⊕ NDArray
// ============================================================

TEST(NDArray_arithmetic, LeftScalar_Float) {
	NDArray a(ArrayList({2.0f, 4.0f, 5.0f}));

	NDArray s = 10.0f + a;
	EXPECT_FLOAT_EQ(s.get<float>({0}), 12.0f);
	EXPECT_FLOAT_EQ(s.get<float>({2}), 15.0f);

	NDArray d = 10.0f - a;
	EXPECT_FLOAT_EQ(d.get<float>({0}), 8.0f);
	EXPECT_FLOAT_EQ(d.get<float>({1}), 6.0f);

	NDArray m = 3.0f * a;
	EXPECT_FLOAT_EQ(m.get<float>({0}), 6.0f);
	EXPECT_FLOAT_EQ(m.get<float>({1}), 12.0f);

	NDArray q = 20.0f / a;
	EXPECT_FLOAT_EQ(q.get<float>({0}), 10.0f);
	EXPECT_FLOAT_EQ(q.get<float>({1}), 5.0f);

	NDArray r = 11.0f % a;
	EXPECT_FLOAT_EQ(r.get<float>({0}), 1.0f); // 11 % 2
	EXPECT_FLOAT_EQ(r.get<float>({1}), 3.0f); // 11 % 4
}

TEST(NDArray_arithmetic, LeftScalar_Int_Div) {
	// The motivating case: int / NDArray
	NDArray a(ArrayList<int32_t>({2, 4, 5}));
	NDArray q = 20 / a;
	EXPECT_EQ(q.type, INT32);
	EXPECT_EQ(q.get<int32_t>({0}), 10);
	EXPECT_EQ(q.get<int32_t>({1}), 5);
	EXPECT_EQ(q.get<int32_t>({2}), 4);

	NDArray r = 11 % a;
	EXPECT_EQ(r.get<int32_t>({0}), 1);
	EXPECT_EQ(r.get<int32_t>({1}), 3);
	EXPECT_EQ(r.get<int32_t>({2}), 1);

	NDArray d = 10 - a;
	EXPECT_EQ(d.get<int32_t>({0}), 8);
	EXPECT_EQ(d.get<int32_t>({2}), 5);
}

TEST(NDArray_arithmetic, LeftScalar_Double) {
	NDArray a(ArrayList({2.0f, 4.0f}));
	NDArray q = 1.5 / a;
	EXPECT_EQ(q.type, F64);
	EXPECT_DOUBLE_EQ(q.get<double>({0}), 0.75);
	EXPECT_DOUBLE_EQ(q.get<double>({1}), 0.375);
}

TEST(NDArray_arithmetic, LeftScalar_Int64) {
	NDArray a(ArrayList<int64_t>({100, 200}));
	NDArray q = (int64_t)1000 / a;
	EXPECT_EQ(q.type, INT64);
	EXPECT_EQ(q.get<int64_t>({0}), 10);
	EXPECT_EQ(q.get<int64_t>({1}), 5);

	NDArray s = (int64_t)3 * a;
	EXPECT_EQ(s.get<int64_t>({0}), 300);
}

TEST(NDArray_arithmetic, LeftScalar_DoesNotMutateRhs) {
	NDArray a(ArrayList({2.0f, 4.0f}));
	NDArray b = 10.0f / a;
	EXPECT_FLOAT_EQ(a.get<float>({0}), 2.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1}), 4.0f);
	EXPECT_FLOAT_EQ(b.get<float>({0}), 5.0f);
}


// ============================================================
// Same-type binaryOp skips convert(); binaryOpInto uses caller dest
//
// CoW / convert rules:
//   convert(same type) always materializes a unique owned snapshot.
//   a+b / binaryOp never convert() a side that already matches the
//   promoted type. Same-type a+b allocates one result buffer.
//   Result is never a CoW alias of either operand (wrap-safe).
//   add / += remain the in-place mutation API.
// ============================================================

TEST(NDArray_arithmetic, SameTypeAdd_AllocatesOnlyResult) {
	alignas(4) float av[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	alignas(4) float bv[4] = {10.0f, 20.0f, 30.0f, 40.0f};
	NDArray a = NDArray::wrap(av, sizeof(av), {4}, F32);
	NDArray b = NDArray::wrap(bv, sizeof(bv), {4}, F32);
	const size_t before = NDArray::ownedBufferAllocCount();
	NDArray c = a + b;
	EXPECT_EQ(NDArray::ownedBufferAllocCount() - before, 1u);
	EXPECT_TRUE(c.ownsStorage());
	EXPECT_NE(c.data(), static_cast<const void*>(av));
	EXPECT_NE(c.data(), static_cast<const void*>(bv));
	EXPECT_FLOAT_EQ(c.get<float>({0}), 11.0f);
	EXPECT_FLOAT_EQ(c.get<float>({3}), 44.0f);
	// wrap operands were only read
	EXPECT_EQ(a.data(), static_cast<const void*>(av));
	EXPECT_EQ(b.data(), static_cast<const void*>(bv));
	EXPECT_FLOAT_EQ(av[0], 1.0f);
	EXPECT_FLOAT_EQ(bv[0], 10.0f);
}

TEST(NDArray_arithmetic, SameTypeMul_AllocatesOnlyResult) {
	NDArray a(ArrayList({2.0f, 3.0f, 4.0f}));
	NDArray b(ArrayList({5.0f, 6.0f, 7.0f}));
	const size_t before = NDArray::ownedBufferAllocCount();
	NDArray c = a.binaryOp(b, NDArray::ArithOp::Mul);
	EXPECT_EQ(NDArray::ownedBufferAllocCount() - before, 1u);
	EXPECT_FLOAT_EQ(c.get<float>({0}), 10.0f);
	EXPECT_FLOAT_EQ(c.get<float>({1}), 18.0f);
	EXPECT_FLOAT_EQ(c.get<float>({2}), 28.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 2.0f);
}

TEST(NDArray_arithmetic, ConvertSameType_StillAllocatesSnapshot) {
	NDArray a(ArrayList({1.0f, 2.0f}));
	const size_t before = NDArray::ownedBufferAllocCount();
	NDArray b = a.convert(F32);
	EXPECT_EQ(NDArray::ownedBufferAllocCount() - before, 1u);
	EXPECT_NE(b.data(), a.data());
	b.set({0}, 9.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 1.0f);
}

TEST(NDArray_arithmetic, BinaryOpInto_WritesCallerBuffer) {
	alignas(4) float out[4] = {0, 0, 0, 0};
	alignas(4) float av[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	alignas(4) float bv[4] = {10.0f, 20.0f, 30.0f, 40.0f};
	NDArray dst = NDArray::wrap(out, sizeof(out), {4}, F32);
	NDArray a = NDArray::wrap(av, sizeof(av), {4}, F32);
	NDArray b = NDArray::wrap(bv, sizeof(bv), {4}, F32);
	const void* p = dst.data();
	const size_t before = NDArray::ownedBufferAllocCount();
	NDArray::binaryOpInto(dst, a, b, NDArray::ArithOp::Add);
	EXPECT_EQ(NDArray::ownedBufferAllocCount(), before);
	EXPECT_EQ(dst.data(), p);
	EXPECT_FALSE(dst.ownsStorage());
	EXPECT_FLOAT_EQ(out[0], 11.0f);
	EXPECT_FLOAT_EQ(out[1], 22.0f);
	EXPECT_FLOAT_EQ(out[2], 33.0f);
	EXPECT_FLOAT_EQ(out[3], 44.0f);
	EXPECT_FLOAT_EQ(av[0], 1.0f);
	EXPECT_FLOAT_EQ(bv[0], 10.0f);
}

TEST(NDArray_arithmetic, BinaryOpInto_SubMulDiv) {
	NDArray a(ArrayList({10.0f, 20.0f}));
	NDArray b(ArrayList({2.0f, 4.0f}));
	NDArray dst({2}, F32);
	const void* p = dst.data();
	NDArray::binaryOpInto(dst, a, b, NDArray::ArithOp::Sub);
	EXPECT_EQ(dst.data(), p);
	EXPECT_FLOAT_EQ(dst.get<float>({0}), 8.0f);
	EXPECT_FLOAT_EQ(dst.get<float>({1}), 16.0f);
	NDArray::binaryOpInto(dst, a, b, NDArray::ArithOp::Mul);
	EXPECT_FLOAT_EQ(dst.get<float>({0}), 20.0f);
	EXPECT_FLOAT_EQ(dst.get<float>({1}), 80.0f);
	NDArray::binaryOpInto(dst, a, b, NDArray::ArithOp::Div);
	EXPECT_FLOAT_EQ(dst.get<float>({0}), 5.0f);
	EXPECT_FLOAT_EQ(dst.get<float>({1}), 5.0f);
}

TEST(NDArray_arithmetic, BinaryOpInto_PromotesIntoMatchingDest) {
	NDArray a({2}, INT8);
	NDArray b(ArrayList({0.5f, 1.5f}));
	a.set({0}, -1);
	a.set({1}, 3);
	NDArray dst({2}, F32);
	const size_t before = NDArray::ownedBufferAllocCount();
	NDArray::binaryOpInto(dst, a, b, NDArray::ArithOp::Add);
	// only the INT8 side is converted; dest is reused
	EXPECT_EQ(NDArray::ownedBufferAllocCount() - before, 1u);
	EXPECT_EQ(dst.type, F32);
	EXPECT_FLOAT_EQ(dst.get<float>({0}), -0.5f);
	EXPECT_FLOAT_EQ(dst.get<float>({1}), 4.5f);
	EXPECT_EQ(a.type, INT8);
	EXPECT_EQ(a.get<int>({0}), -1);
}

TEST(NDArray_arithmetic, BinaryOpInto_WrongDestTypeThrows) {
	NDArray a(ArrayList({1.0f, 2.0f}));
	NDArray b(ArrayList({3.0f, 4.0f}));
	NDArray dst({2}, F64);
	EXPECT_THROW(NDArray::binaryOpInto(dst, a, b, NDArray::ArithOp::Add), std::invalid_argument);
}

TEST(NDArray_arithmetic, BinaryOpInto_AliasingDestIsA) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f}));
	NDArray b(ArrayList({4.0f, 5.0f, 6.0f}));
	NDArray::binaryOpInto(a, a, b, NDArray::ArithOp::Add);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 5.0f);
	EXPECT_FLOAT_EQ(a.get<float>({1}), 7.0f);
	EXPECT_FLOAT_EQ(a.get<float>({2}), 9.0f);
}

TEST(NDArray_arithmetic, BinaryOpInto_AliasingDestIsB_NonCommutative) {
	NDArray a(ArrayList({10.0f, 20.0f}));
	NDArray b(ArrayList({3.0f, 5.0f}));
	NDArray::binaryOpInto(b, a, b, NDArray::ArithOp::Sub);
	EXPECT_FLOAT_EQ(b.get<float>({0}), 7.0f);
	EXPECT_FLOAT_EQ(b.get<float>({1}), 15.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 10.0f);
}

TEST(NDArray_arithmetic, MixedTypeAdd_StillPromotes) {
	NDArray a({2}, INT8);
	a.set({0}, -1);
	a.set({1}, 2);
	NDArray b({2}, INT32);
	b.set({0}, 10);
	b.set({1}, 20);
	NDArray c = a + b;
	EXPECT_EQ(c.type, INT32);
	EXPECT_EQ(c.get<int32_t>({0}), 9);
	EXPECT_EQ(c.get<int32_t>({1}), 22);
	EXPECT_EQ(a.type, INT8);
	EXPECT_EQ(a.get<int>({0}), -1);
}
