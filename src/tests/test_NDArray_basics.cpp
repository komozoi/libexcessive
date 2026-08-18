
// Copyright 2021-2026 komozoi
// Original Creation Date: 2026-8-2
//
// All rights reserved.  Do not copy, distribute, or execute, in compiled or source form,
// any portion of this software, except software not created by me.
// If you find this software, please notify me immediately: komozoi@protonmail.com
//
//

#include <gtest/gtest.h>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "NDArray.h"


TEST(NDArray_basics, Construct) {
	NDArray vector(ArrayList({1.0f, 2.0f, 3.0f}));

	EXPECT_EQ(vector.type, F32);
	ASSERT_EQ(vector.shape.size(), 1);
	ASSERT_EQ(vector.shape.get(0), 3);
	EXPECT_EQ(vector.get<float>({0}), 1.0f);
}


TEST(NDArray_basics, Update) {
	NDArray vector(ArrayList({1.0f, 2.0f, 3.0f}));

	EXPECT_EQ(vector.get<float>({0}), 1.0f);

	vector.set({0}, 4.5f);
	EXPECT_EQ(vector.get<float>({0}), 4.5f);
}


TEST(NDArray_basics, Binary) {
	NDArray vector({16}, BINARY);

	EXPECT_EQ(vector.type, BINARY);
	ASSERT_EQ(vector.shape.size(), 1);
	ASSERT_EQ(vector.shape.get(0), 16);
	EXPECT_EQ(vector.get<uint8_t>({0}), 0);

	vector.set({0}, 4.5f);
	EXPECT_EQ(vector.get<float>({0}), 1.0f);
	EXPECT_EQ(vector.get<int>({0}), 1);
	EXPECT_EQ(vector.get<uint256_t>({0}), uint256_t(1));

	vector.set({0}, -55.0f);
	EXPECT_EQ(vector.get<float>({0}), 0.0f);
	EXPECT_EQ(vector.get<int>({0}), 0);
	EXPECT_EQ(vector.get<uint256_t>({0}), uint256_t(0));

	vector.set({0}, 0.0f);
	EXPECT_EQ(vector.get<float>({0}), 0.0f);
	EXPECT_EQ(vector.get<int>({0}), 0);
	EXPECT_EQ(vector.get<uint256_t>({0}), uint256_t(0));
}


// ============================================================
// Rank / shape validation
// ============================================================

TEST(NDArray_basics, RankMismatch_Throws) {
    NDArray a({2, 3}, F32);

    EXPECT_THROW(a.get<float>({0}), std::out_of_range);
    EXPECT_THROW(a.get<float>({0, 1, 2}), std::out_of_range);
    EXPECT_THROW(a.set<float>({0}, 1.0f), std::out_of_range);
    EXPECT_THROW(a.set<float>({0, 1, 2}, 1.0f), std::out_of_range);
}

TEST(NDArray_basics, EmptyIndicesOnScalar) {
    // 0-dimensional (scalar)
    NDArray s({}, F32);
    s.set({}, 3.14f);
    EXPECT_FLOAT_EQ(s.get<float>({}), 3.14f);
    EXPECT_EQ(s.get<int>({}), 3);
}

TEST(NDArray_basics, ScalarVsVector) {
    NDArray scalar({}, F32);
    NDArray vector({1}, F32);

    scalar.set({}, 7.0f);
    vector.set({0}, 7.0f);

    EXPECT_FLOAT_EQ(scalar.get<float>({}), 7.0f);
    EXPECT_FLOAT_EQ(vector.get<float>({0}), 7.0f);
}

TEST(NDArray_basics, ScalarConvenienceConstructors) {
	NDArray s1(F32, 3.14f);
	EXPECT_FLOAT_EQ(s1.get<float>({}), 3.14f);
	EXPECT_EQ(s1.shape.size(), 0);

	NDArray s2(UINT8, 42);
	EXPECT_EQ(s2.get<uint8_t>({}), 42);
	EXPECT_EQ(s2.shape.size(), 0);
}

// ============================================================
// Bounds checking (individual indices)
// ============================================================

TEST(NDArray_basics, OutOfBounds_Throws) {
    NDArray a({3, 4}, F32);

    // These should throw once proper per-axis bounds checks exist
    EXPECT_THROW(a.get<float>({-1, 0}), std::out_of_range);
    EXPECT_THROW(a.get<float>({0, -1}), std::out_of_range);
    EXPECT_THROW(a.get<float>({3, 0}), std::out_of_range);
    EXPECT_THROW(a.get<float>({0, 4}), std::out_of_range);
    EXPECT_THROW(a.get<float>({3, 4}), std::out_of_range);

    EXPECT_THROW(a.set<float>({-1, 0}, 1.0f), std::out_of_range);
    EXPECT_THROW(a.set<float>({0, 4}, 1.0f), std::out_of_range);
}

// ============================================================
// Multi-dimensional access
// ============================================================

TEST(NDArray_basics, Matrix_F32) {
    NDArray m({2, 3}, F32);

    m.set({0, 0}, 1.0f);
    m.set({0, 1}, 2.0f);
    m.set({0, 2}, 3.0f);
    m.set({1, 0}, 4.0f);
    m.set({1, 1}, 5.0f);
    m.set({1, 2}, 6.0f);

    EXPECT_FLOAT_EQ(m.get<float>({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(m.get<float>({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(m.get<float>({0, 2}), 3.0f);
    EXPECT_FLOAT_EQ(m.get<float>({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(m.get<float>({1, 1}), 5.0f);
    EXPECT_FLOAT_EQ(m.get<float>({1, 2}), 6.0f);
}

TEST(NDArray_basics, Tensor3D_F32) {
    NDArray t({2, 2, 2}, F32);

    float v = 0.0f;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
                t.set({i, j, k}, v++);

    EXPECT_FLOAT_EQ(t.get<float>({0, 0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(t.get<float>({0, 0, 1}), 1.0f);
    EXPECT_FLOAT_EQ(t.get<float>({0, 1, 0}), 2.0f);
    EXPECT_FLOAT_EQ(t.get<float>({0, 1, 1}), 3.0f);
    EXPECT_FLOAT_EQ(t.get<float>({1, 0, 0}), 4.0f);
    EXPECT_FLOAT_EQ(t.get<float>({1, 0, 1}), 5.0f);
    EXPECT_FLOAT_EQ(t.get<float>({1, 1, 0}), 6.0f);
    EXPECT_FLOAT_EQ(t.get<float>({1, 1, 1}), 7.0f);
}

// ============================================================
// BINARY specific (including the shift bug)
// ============================================================

TEST(NDArray_basics, BasicBitSetClear) {
    NDArray b({64}, BINARY);

    // All should start at 0
    for (int i = 0; i < 64; ++i)
        EXPECT_EQ(b.get<int>({i}), 0);

    // Set bit 0
    b.set({0}, 1);
    EXPECT_EQ(b.get<int>({0}), 1);
    EXPECT_EQ(b.get<float>({0}), 1.0f);

    // Clear it
    b.set({0}, 0);
    EXPECT_EQ(b.get<int>({0}), 0);

    // Set a high bit (this is where `1 <<` instead of `1ULL <<` blows up)
    b.set({63}, 1);
    EXPECT_EQ(b.get<int>({63}), 1);
    EXPECT_EQ(b.get<int>({0}), 0);   // must not have corrupted bit 0
}

TEST(NDArray_basics, MultipleWords) {
    // 128 bits → two uint64_t
    NDArray b({128}, BINARY);

    b.set({0}, 1);
    b.set({63}, 1);
    b.set({64}, 1);
    b.set({127}, 1);

    EXPECT_EQ(b.get<int>({0}), 1);
    EXPECT_EQ(b.get<int>({63}), 1);
    EXPECT_EQ(b.get<int>({64}), 1);
    EXPECT_EQ(b.get<int>({127}), 1);

    // Everything else still 0
    EXPECT_EQ(b.get<int>({1}), 0);
    EXPECT_EQ(b.get<int>({62}), 0);
    EXPECT_EQ(b.get<int>({65}), 0);
    EXPECT_EQ(b.get<int>({126}), 0);
}

TEST(NDArray_basics, Truthiness) {
    NDArray b({8}, BINARY);

    b.set({0}, 0.0f);
    b.set({1}, 0.1f);   // any positive → 1
    b.set({2}, -3.0f);  // negative → 0
    b.set({3}, 100.0f);

    EXPECT_EQ(b.get<int>({0}), 0);
    EXPECT_EQ(b.get<int>({1}), 1);
    EXPECT_EQ(b.get<int>({2}), 0);
    EXPECT_EQ(b.get<int>({3}), 1);
}

// ============================================================
// Type conversion / templated get
// ============================================================

TEST(NDArray_basics, F32_Conversions) {
    NDArray a({3}, F32);
    a.set({0}, 3.7f);
    a.set({1}, -2.2f);
    a.set({2}, 0.0f);

    EXPECT_FLOAT_EQ(a.get<float>({0}), 3.7f);
    EXPECT_EQ(a.get<int>({0}), 3);          // truncation toward zero
    EXPECT_EQ(a.get<int>({1}), -2);
    EXPECT_EQ(a.get<uint8_t>({0}), 3);
}

TEST(NDArray_basics, UINT8_Conversions) {
    NDArray a({4}, UINT8);
    a.set({0}, 0);
    a.set({1}, 255);
    a.set({2}, 128);
    a.set({3}, 1.9f);   // should truncate

    EXPECT_EQ(a.get<uint8_t>({0}), 0);
    EXPECT_EQ(a.get<uint8_t>({1}), 255);
    EXPECT_EQ(a.get<int>({1}), 255);
    EXPECT_EQ(a.get<float>({1}), 255.0f);
    EXPECT_EQ(a.get<uint8_t>({3}), 1);
}

TEST(NDArray_basics, UINT256_Basic) {
    NDArray a({2}, UINT256);

    a.set({0}, 0);
    a.set({1}, 42);

    EXPECT_EQ(a.get<uint256_t>({0}), uint256_t(0));
    EXPECT_EQ(a.get<uint256_t>({1}), uint256_t(42));
    EXPECT_EQ(a.get<int>({1}), 42);
    EXPECT_FLOAT_EQ(a.get<float>({1}), 42.0f);
}

// ============================================================
// Construction edge cases
// ============================================================

TEST(NDArray_basics, EmptyShape_F32) {
    NDArray s({}, F32);
    EXPECT_EQ(s.shape.size(), 0);
    EXPECT_EQ(s.type, F32);
    // should be able to get/set the single scalar value
    s.set({}, 1.5f);
    EXPECT_FLOAT_EQ(s.get<float>({}), 1.5f);
}

TEST(NDArray_basics, EmptyShape_BINARY) {
    NDArray s({}, BINARY);
    EXPECT_EQ(s.shape.size(), 0);
    s.set({}, 1);
    EXPECT_EQ(s.get<int>({}), 1);
}

TEST(NDArray_basics, FromFloatVector) {
    NDArray v(ArrayList({1.0f, 2.0f, 3.0f, 4.0f}));
    EXPECT_EQ(v.type, F32);
    ASSERT_EQ(v.shape.size(), 1);
    EXPECT_EQ(v.shape.get(0), 4);
    EXPECT_FLOAT_EQ(v.get<float>({0}), 1.0f);
    EXPECT_FLOAT_EQ(v.get<float>({3}), 4.0f);
}

TEST(NDArray_basics, FromUint8Vector) {
    NDArray v(ArrayList<uint8_t>({10, 20, 30}));
    EXPECT_EQ(v.type, UINT8);
    ASSERT_EQ(v.shape.size(), 1);
    EXPECT_EQ(v.shape.get(0), 3);
    EXPECT_EQ(v.get<uint8_t>({0}), 10);
    EXPECT_EQ(v.get<uint8_t>({2}), 30);
}

TEST(NDArray_basics, ShapePlusData_F32) {
    NDArray m({2, 2}, ArrayList({1.0f, 2.0f, 3.0f, 4.0f}));
    EXPECT_EQ(m.type, F32);
    ASSERT_EQ(m.shape.size(), 2);
    EXPECT_EQ(m.shape.get(0), 2);
    EXPECT_EQ(m.shape.get(1), 2);
    EXPECT_FLOAT_EQ(m.get<float>({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(m.get<float>({1, 1}), 4.0f);
}

TEST(NDArray_basics, ZeroAxis_IsEmpty) {
	NDArray a({0, 8}, F32);
	EXPECT_EQ(a.numElements(), (size_t)0);
	EXPECT_EQ(a.shape.get(0), 0);
	EXPECT_EQ(a.shape.get(1), 8);
	EXPECT_TRUE(a.isEmpty());
	EXPECT_FALSE(a.ownsStorage());
	EXPECT_EQ(a.view().sharedBuffer().get(), nullptr);
}

// ============================================================
// Default / empty construction (no dummy buffer)
// ============================================================

static void expectEmptyNoBuffer(const NDArray& a, NDArrayType type, const ArrayList<int>& shape) {
	EXPECT_TRUE(a.isEmpty());
	EXPECT_EQ(a.numElements(), (size_t)0);
	EXPECT_EQ(a.type, type);
	ASSERT_EQ(a.shape.size(), shape.size());
	for (int i = 0; i < shape.size(); ++i)
		EXPECT_EQ(a.shape.get(i), shape.get(i));
	EXPECT_FALSE(a.ownsStorage());
	EXPECT_EQ(a.view().sharedBuffer().get(), nullptr);
}

TEST(NDArray_basics, DefaultConstructible) {
	static_assert(std::is_default_constructible_v<NDArray>, "NDArray must be default-constructible");
	NDArray a;
	expectEmptyNoBuffer(a, F32, {0});
}

TEST(NDArray_basics, EmptyFactory) {
	expectEmptyNoBuffer(NDArray::empty(), F32, {0});
	expectEmptyNoBuffer(NDArray::empty(INT3), INT3, {0});
	expectEmptyNoBuffer(NDArray::empty(BINARY), BINARY, {0});
}

TEST(NDArray_basics, EmptyAsStructMember) {
	struct Batch {
		NDArray result;
		int id = 0;
	};
	static_assert(std::is_default_constructible_v<Batch>, "struct with NDArray member");
	Batch b;
	EXPECT_EQ(b.id, 0);
	expectEmptyNoBuffer(b.result, F32, {0});
}

TEST(NDArray_basics, EmptyAssignAndMove) {
	NDArray empty;
	NDArray filled({3}, F32);
	filled.set({0}, 1.0f);
	filled.set({1}, 2.0f);
	filled.set({2}, 3.0f);

	NDArray a = empty;
	expectEmptyNoBuffer(a, F32, {0});
	EXPECT_TRUE(a == empty);

	a = filled;
	EXPECT_FALSE(a.isEmpty());
	EXPECT_EQ(a.numElements(), (size_t)3);
	EXPECT_FLOAT_EQ(a.get<float>({1}), 2.0f);
	EXPECT_TRUE(a.ownsStorage());

	a = empty;
	expectEmptyNoBuffer(a, F32, {0});

	NDArray movedFromFilled = std::move(filled);
	EXPECT_EQ(movedFromFilled.numElements(), (size_t)3);
	EXPECT_FLOAT_EQ(movedFromFilled.get<float>({2}), 3.0f);
	expectEmptyNoBuffer(filled, F32, {0});

	NDArray dest({2}, UINT8);
	dest = std::move(empty);
	expectEmptyNoBuffer(dest, F32, {0});
	expectEmptyNoBuffer(empty, F32, {0});

	NDArray src = NDArray::empty(INT3);
	NDArray taken = std::move(src);
	expectEmptyNoBuffer(taken, INT3, {0});
	expectEmptyNoBuffer(src, INT3, {0});
}

TEST(NDArray_basics, EmptyScalarIsNotEmpty) {
	NDArray s({}, F32);
	EXPECT_FALSE(s.isEmpty());
	EXPECT_EQ(s.numElements(), (size_t)1);
	EXPECT_TRUE(s.ownsStorage());
	EXPECT_NE(s.view().sharedBuffer().get(), nullptr);
}

TEST(NDArray_basics, EmptyConvertAndIndexThrow) {
	NDArray a = NDArray::empty(UINT8);
	NDArray b = a.convert(F32);
	expectEmptyNoBuffer(b, F32, {0});
	EXPECT_THROW(a.get<uint8_t>({0}), std::out_of_range);
	EXPECT_THROW(a.set({0}, 1), std::out_of_range);
	EXPECT_THROW(a.getFlat<uint8_t>(0), std::out_of_range);
	EXPECT_THROW((void)a.sum(), std::invalid_argument);
}

TEST(NDArray_basics, EmptyFromEmptyVector) {
	NDArray v(ArrayList<float>{});
	expectEmptyNoBuffer(v, F32, {0});
}

TEST(NDArray_basics, NegativeAxis_Throws) {
	EXPECT_THROW(NDArray({-1, 8}, F32), std::invalid_argument);
	EXPECT_THROW(NDArray({8, -2}, UINT8), std::invalid_argument);
}

TEST(NDArray_basics, ShapeProductOverflow_Throws) {
	// 2^30 * 2^30 * 2^10 overflows size_t
	EXPECT_THROW(NDArray({1 << 30, 1 << 30, 1 << 10}, UINT8), std::invalid_argument);
}

TEST(NDArray_basics, Uint256ByteSizeOverflow_Throws) {
	// 2^60 elements fit in size_t; 2^60 * 32 wraps
	EXPECT_THROW(NDArray({1 << 30, 1 << 30}, UINT256), std::invalid_argument);
}

// ============================================================
// Write / read round-trips & overwrite
// ============================================================

TEST(NDArray_basics, Overwrite) {
    NDArray a({5}, F32);

    for (int i = 0; i < 5; ++i)
        a.set({i}, float(i));

    // overwrite
    a.set({2}, 99.0f);
    a.set({4}, -1.5f);

    EXPECT_FLOAT_EQ(a.get<float>({0}), 0.0f);
    EXPECT_FLOAT_EQ(a.get<float>({1}), 1.0f);
    EXPECT_FLOAT_EQ(a.get<float>({2}), 99.0f);
    EXPECT_FLOAT_EQ(a.get<float>({3}), 3.0f);
    EXPECT_FLOAT_EQ(a.get<float>({4}), -1.5f);
}

TEST(NDArray_basics, LargeOffset_BINARY) {
    // Forces use of the second (and third) uint64 word
    NDArray b({200}, BINARY);

    b.set({0}, 1);
    b.set({64}, 1);
    b.set({128}, 1);
    b.set({199}, 1);

    EXPECT_EQ(b.get<int>({0}), 1);
    EXPECT_EQ(b.get<int>({64}), 1);
    EXPECT_EQ(b.get<int>({128}), 1);
    EXPECT_EQ(b.get<int>({199}), 1);

    // neighbours stay zero
    EXPECT_EQ(b.get<int>({1}), 0);
    EXPECT_EQ(b.get<int>({63}), 0);
    EXPECT_EQ(b.get<int>({65}), 0);
    EXPECT_EQ(b.get<int>({127}), 0);
    EXPECT_EQ(b.get<int>({129}), 0);
    EXPECT_EQ(b.get<int>({198}), 0);
}

// ============================================================
// Equality (once operator== is implemented)
// ============================================================

TEST(NDArray_basics, SameContent) {
    NDArray a({2, 2}, F32);
    NDArray b({2, 2}, F32);

    a.set({0, 0}, 1.0f); a.set({0, 1}, 2.0f);
    a.set({1, 0}, 3.0f); a.set({1, 1}, 4.0f);

    b.set({0, 0}, 1.0f); b.set({0, 1}, 2.0f);
    b.set({1, 0}, 3.0f); b.set({1, 1}, 4.0f);

    EXPECT_TRUE(a == b);
}

TEST(NDArray_basics, DifferentShape) {
    NDArray a({2, 3}, F32);
    NDArray b({3, 2}, F32);
    EXPECT_FALSE(a == b);
}

TEST(NDArray_basics, DifferentType) {
    NDArray a({4}, F32);
    NDArray b({4}, UINT8);
    EXPECT_FALSE(a == b);
}

TEST(NDArray_basics, DifferentData) {
    NDArray a({3}, F32);
    NDArray b({3}, F32);
    a.set({0}, 1.0f);
    b.set({0}, 2.0f);
    EXPECT_FALSE(a == b);
}


// ============================================================
// bool set (instantiates setAtOffset<bool> — must compile under -Werror)
// ============================================================

TEST(NDArray_basics, SetBool_OnBinaryAndIntTypes) {
	// Beta report: setAtOffset<bool> hit `value < 0` under UINT256 branch
	// (-Werror=bool-compare) even when the runtime type is BINARY.
	NDArray b({4}, BINARY);
	b.set({0}, true);
	b.set({1}, false);
	b.set({2}, true);
	b.set({3}, false);
	EXPECT_EQ(b.get<int>({0}), 1);
	EXPECT_EQ(b.get<int>({1}), 0);
	EXPECT_EQ(b.get<int>({2}), 1);
	EXPECT_EQ(b.get<int>({3}), 0);

	NDArray i({2}, INT32);
	i.set({0}, true);
	i.set({1}, false);
	EXPECT_EQ(i.get<int>({0}), 1);
	EXPECT_EQ(i.get<int>({1}), 0);

	NDArray u({2}, UINT256);
	u.set({0}, true);
	u.set({1}, false);
	EXPECT_EQ(u.get<uint256_t>({0}), uint256_t(1));
	EXPECT_EQ(u.get<uint256_t>({1}), uint256_t(0));
}
