
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
// Explicit convert() — the only API that may drop precision
// ============================================================

TEST(NDArray_conversions, SameType_IsCopy) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f}));
	NDArray b = a.convert(F32);

	EXPECT_EQ(b.type, F32);
	EXPECT_EQ(b.shape.size(), 1);
	EXPECT_EQ(b.shape.get(0), 3);
	EXPECT_FLOAT_EQ(b.get<float>({0}), 1.0f);
	EXPECT_FLOAT_EQ(b.get<float>({2}), 3.0f);

	// Independent storage
	b.set({0}, 99.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 1.0f);
}

TEST(NDArray_conversions, UINT8_to_F32_Lossless) {
	NDArray a(ArrayList<uint8_t>({0, 1, 127, 255}));
	NDArray b = a.convert(F32);

	EXPECT_EQ(b.type, F32);
	EXPECT_FLOAT_EQ(b.get<float>({0}), 0.0f);
	EXPECT_FLOAT_EQ(b.get<float>({1}), 1.0f);
	EXPECT_FLOAT_EQ(b.get<float>({2}), 127.0f);
	EXPECT_FLOAT_EQ(b.get<float>({3}), 255.0f);
}

TEST(NDArray_conversions, UINT8_to_UINT256_Lossless) {
	NDArray a(ArrayList<uint8_t>({0, 42, 255}));
	NDArray b = a.convert(UINT256);

	EXPECT_EQ(b.type, UINT256);
	EXPECT_EQ(b.get<uint256_t>({0}), uint256_t(0));
	EXPECT_EQ(b.get<uint256_t>({1}), uint256_t(42));
	EXPECT_EQ(b.get<uint256_t>({2}), uint256_t(255));
}

TEST(NDArray_conversions, BINARY_to_UINT8) {
	NDArray a({4}, BINARY);
	a.set({0}, 1);
	a.set({1}, 0);
	a.set({2}, 1);
	a.set({3}, 0);

	NDArray b = a.convert(UINT8);
	EXPECT_EQ(b.type, UINT8);
	EXPECT_EQ(b.get<uint8_t>({0}), 1);
	EXPECT_EQ(b.get<uint8_t>({1}), 0);
	EXPECT_EQ(b.get<uint8_t>({2}), 1);
	EXPECT_EQ(b.get<uint8_t>({3}), 0);
}

TEST(NDArray_conversions, BINARY_to_F32) {
	NDArray a({3}, BINARY);
	a.set({0}, 1);
	a.set({1}, 0);
	a.set({2}, 1);

	NDArray b = a.convert(F32);
	EXPECT_EQ(b.type, F32);
	EXPECT_FLOAT_EQ(b.get<float>({0}), 1.0f);
	EXPECT_FLOAT_EQ(b.get<float>({1}), 0.0f);
	EXPECT_FLOAT_EQ(b.get<float>({2}), 1.0f);
}

TEST(NDArray_conversions, F32_to_UINT8_Truncates) {
	// Programmer asked for lossy conversion
	NDArray a(ArrayList({3.7f, 255.9f, 0.1f}));
	NDArray b = a.convert(UINT8);

	EXPECT_EQ(b.type, UINT8);
	EXPECT_EQ(b.get<uint8_t>({0}), 3);
	EXPECT_EQ(b.get<uint8_t>({1}), 255);
	EXPECT_EQ(b.get<uint8_t>({2}), 0);
}

TEST(NDArray_conversions, F32_to_BINARY_Truthiness) {
	NDArray a(ArrayList({0.0f, 0.1f, -1.0f, 5.0f}));
	NDArray b = a.convert(BINARY);

	EXPECT_EQ(b.type, BINARY);
	EXPECT_EQ(b.get<int>({0}), 0);
	EXPECT_EQ(b.get<int>({1}), 1);  // positive → 1
	EXPECT_EQ(b.get<int>({2}), 0);  // non-positive → 0
	EXPECT_EQ(b.get<int>({3}), 1);
}

TEST(NDArray_conversions, UINT256_to_UINT8_Wraps) {
	NDArray a({2}, UINT256);
	a.set({0}, 42);
	a.set({1}, 300); // 300 & 0xFF = 44

	NDArray b = a.convert(UINT8);
	EXPECT_EQ(b.get<uint8_t>({0}), 42);
	EXPECT_EQ(b.get<uint8_t>({1}), (uint8_t)300);
}

TEST(NDArray_conversions, UINT256_to_F32) {
	NDArray a({2}, UINT256);
	a.set({0}, 0);
	a.set({1}, 100);

	NDArray b = a.convert(F32);
	EXPECT_EQ(b.type, F32);
	EXPECT_FLOAT_EQ(b.get<float>({0}), 0.0f);
	EXPECT_FLOAT_EQ(b.get<float>({1}), 100.0f);
}

TEST(NDArray_conversions, Scalar_PreservesEmptyShape) {
	NDArray s(F32, 3.5f);
	NDArray t = s.convert(UINT8);

	EXPECT_EQ(t.shape.size(), 0);
	EXPECT_EQ(t.type, UINT8);
	EXPECT_EQ(t.get<uint8_t>({}), 3);
}

TEST(NDArray_conversions, Matrix_ShapePreserved) {
	NDArray m({2, 2}, ArrayList({1.0f, 2.0f, 3.0f, 4.0f}));
	NDArray u = m.convert(UINT8);

	ASSERT_EQ(u.shape.size(), 2);
	EXPECT_EQ(u.shape.get(0), 2);
	EXPECT_EQ(u.shape.get(1), 2);
	EXPECT_EQ(u.get<uint8_t>({0, 0}), 1);
	EXPECT_EQ(u.get<uint8_t>({1, 1}), 4);
}

TEST(NDArray_conversions, RoundTrip_UINT8_F32_UINT8) {
	NDArray a(ArrayList<uint8_t>({0, 10, 200, 255}));
	NDArray b = a.convert(F32).convert(UINT8);

	EXPECT_EQ(b.get<uint8_t>({0}), 0);
	EXPECT_EQ(b.get<uint8_t>({1}), 10);
	EXPECT_EQ(b.get<uint8_t>({2}), 200);
	EXPECT_EQ(b.get<uint8_t>({3}), 255);
}
